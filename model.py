import os
import torch
import torch.nn as nn
from utils import init_net
import torch.nn.functional as F


import networks
import losses
import utils
from torchvision.models import vgg19
vgg_activation = dict()

def get_activation(name):
    def hook(model, input, output):
        vgg_activation[name] = output

    return hook

def gram_matrix(x):
    a, b, c, d = x.size()
    features = x.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class CSDMT(nn.Module):
    def __init__(self, opts):
        super(CSDMT, self).__init__()
        # parameters
        self.opts = opts
        self.lr = opts.lr
        self.batch_size = opts.batch_size
        self.gpu = torch.device('cuda:{}'.format(opts.gpu)) if opts.gpu >= 0 else torch.device('cpu')

        self.weight_semantic = opts.weight_semantic
        self.weight_corr = opts.weight_corr
        self.weight_identity = opts.weight_identity
        self.weight_back = opts.weight_back
        self.weight_adv = opts.weight_adv
        self.weight_contrastive=opts.weight_contrastive

        self.weight_self_recL1 = opts.weight_self_recL1
        self.weight_cycleL1 = opts.weight_cycleL1
        self.target_layer=['relu_3', 'relu_8']


        if self.opts.phase == 'train':
            self.vgg = vgg19(pretrained=True)
            for layer in self.target_layer:
                self.vgg.features[int(layer.split('_')[-1])].register_forward_hook(get_activation(layer))
            self.vgg.to(self.gpu)
            self.FloatTensor = torch.cuda.FloatTensor if opts.gpu >= 0 else torch.FloatTensor
            self.criterionL1 = nn.L1Loss()
            self.criterionL2 = nn.MSELoss()
            self.criterionIdentity = losses.GPLoss()
            self.criterionGAN = losses.GANLoss(gan_mode=opts.gan_mode, tensor=self.FloatTensor)


        self._build_model()

    def _build_model(self):

        print('start build model')
        # discriminators
        if self.opts.dis_scale > 1:
            self.dis = init_net(networks.MultiScaleDis(3, self.opts.dis_scale, norm=self.opts.dis_norm,
                                                        sn=self.opts.dis_sn), self.gpu, init_type='normal', gain=0.02)
        else:
            self.dis = init_net(networks.Dis(3, norm=self.opts.dis_norm, sn=self.opts.dis_sn), self.gpu,
                                init_type='normal', gain=0.02)
        self.gen = init_net(networks.Generator(input_dim=3,parse_dim=self.opts.semantic_dim,ngf=16), self.gpu,
                                     init_type='normal', gain=0.02)


        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=self.lr, betas=(0.5, 0.999),
                                        weight_decay=0.0001)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.5, 0.999),
                                                 weight_decay=0.0001)

        print('finish build model')

    def load_data(self, data):
        self.non_makeup_color_img = data['non_makeup_color_img'].to(self.gpu).detach()
        self.non_makeup_split_parse = data['non_makeup_split_parse'].to(self.gpu).detach()
        self.non_makeup_all_mask = data['non_makeup_all_mask'].to(self.gpu).detach()
        self.non_makeup_face_mask = data['non_makeup_face_mask'].to(self.gpu).detach()
        self.non_makeup_brow_mask = data['non_makeup_brow_mask'].to(self.gpu).detach()
        self.non_makeup_eye_mask = data['non_makeup_eye_mask'].to(self.gpu).detach()
        self.non_makeup_lip_mask = data['non_makeup_lip_mask'].to(self.gpu).detach()
        self.non_makeup_face_mask_no_neck_no_ear = self.non_makeup_face_mask+self.non_makeup_brow_mask+self.non_makeup_eye_mask+self.non_makeup_lip_mask

        self.makeup_color_img = data['makeup_color_img'].to(self.gpu).detach()
        self.makeup_split_parse = data['makeup_split_parse'].to(self.gpu).detach()
        self.makeup_all_mask = data['makeup_all_mask'].to(self.gpu).detach()
        self.makeup_face_mask = data['makeup_face_mask'].to(self.gpu).detach()
        self.makeup_brow_mask = data['makeup_brow_mask'].to(self.gpu).detach()
        self.makeup_eye_mask = data['makeup_eye_mask'].to(self.gpu).detach()
        self.makeup_lip_mask = data['makeup_lip_mask'].to(self.gpu).detach()
        self.makeup_face_mask_no_neck_no_ear = self.makeup_face_mask + self.makeup_brow_mask + self.makeup_eye_mask + self.makeup_lip_mask

        self.makeup_color_change_img = data['makeup_color_change_img'].to(self.gpu).detach()
        self.makeup_color_change_warp_img = data['makeup_color_change_warp_img'].to(self.gpu).detach()
        self.makeup_split_warp_parse = data['makeup_split_parse_warp'].to(self.gpu).detach()
        self.makeup_all_warp_mask = data['makeup_all_warp_mask'].to(self.gpu).detach()
        self.makeup_face_warp_mask = data['makeup_face_warp_mask'].to(self.gpu).detach()
        self.makeup_brow_warp_mask = data['makeup_brow_warp_mask'].to(self.gpu).detach()
        self.makeup_eye_warp_mask = data['makeup_eye_warp_mask'].to(self.gpu).detach()
        self.makeup_lip_warp_mask = data['makeup_lip_warp_mask'].to(self.gpu).detach()
        self.makeup_face_warp_mask_no_neck_no_ear = self.makeup_face_warp_mask + self.makeup_brow_warp_mask + self.makeup_eye_warp_mask + self.makeup_lip_warp_mask

        self.makeup_color_change2_img = data['makeup_color_change_img2'].to(self.gpu).detach()
        self.makeup_color_change3_img = data['makeup_color_change_img3'].to(self.gpu).detach()
        self.makeup_color_change4_img = data['makeup_color_change_img4'].to(self.gpu).detach()

    # def warp(self,x,map):
    #     n, c, h, w = x.shape
    #     x_warp = torch.bmm(x.view(n, c, h * w), map)  # n*HW*1
    #     x_warp = x_warp.view(n, c, h, w)
    #     return x_warp


    def forward(self):

        # makeup transfer
        self.transfer_output_data=self.gen(source_img=self.non_makeup_color_img,
                                           source_parse=self.non_makeup_split_parse,
                                           source_all_mask=self.non_makeup_all_mask,
                                           ref_img=self.makeup_color_img,
                                           ref_parse=self.makeup_split_parse,
                                           ref_all_mask=self.makeup_all_mask)
        # cycle consistent
        self.cycle_output_data=self.gen(source_img=self.makeup_color_img,
                                           source_parse=self.makeup_split_parse,
                                           source_all_mask=self.makeup_all_mask,
                                           ref_img=self.transfer_output_data['transfer_img'],
                                           ref_parse=self.non_makeup_split_parse,
                                           ref_all_mask=self.non_makeup_all_mask)

        # self supervision
        self.rec_output_data=self.gen(source_img=self.makeup_color_change_img,
                                           source_parse=self.makeup_split_parse,
                                           source_all_mask=self.makeup_all_mask,
                                           ref_img=self.makeup_color_change_warp_img,
                                           ref_parse=self.non_makeup_split_parse,
                                           ref_all_mask=self.makeup_all_warp_mask,
                                      replace_content_img=self.makeup_color_img)

    def update_D(self):
        self.forward()
        self.dis_opt.zero_grad()
        loss_dis = self.backward_D(self.dis, self.makeup_color_img, self.transfer_output_data['transfer_img'])
        self.loss_dis = loss_dis.item()
        self.dis_opt.step()

    def backward_D(self, netD, real, fake):
        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)
        ad_true_loss = self.criterionGAN(pred_real, target_is_real=True, for_discriminator=True)
        ad_fake_loss = self.criterionGAN(pred_fake, target_is_real=False, for_discriminator=True)
        loss_D = ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D

    def update_G(self):
        self.gen_opt.zero_grad()
        self.backward_G()
        self.gen_opt.step()

    def warp_basedon_corr(self, x, corr):
        n, c, h, w = x.shape
        x_warp = torch.bmm(x.view(n, c, h * w), corr)  # n*HW*1
        x_warp = x_warp.view(n, c, h, w)
        return x_warp

    def gram_style(self, reference, fake_images):
        fake_activation = dict()
        real_activation = dict()

        # percep_style
        g_loss_style = 0

        self.vgg(reference)
        for layer in self.target_layer:
            real_activation[layer] = vgg_activation[layer]
        vgg_activation.clear()

        self.vgg(fake_images)
        for layer in self.target_layer:
            fake_activation[layer] = vgg_activation[layer]
        vgg_activation.clear()

        for layer in self.target_layer:
            g_loss_style += self.criterionL1(gram_matrix(fake_activation[layer]), gram_matrix(real_activation[layer]))

        return g_loss_style

    def backward_G(self):
        #corr
        non_makeup_split_parse_down4 = F.interpolate(self.non_makeup_split_parse, scale_factor=1 / 4, mode='nearest')

        makeup_split_parse_down4 = F.interpolate(self.makeup_split_parse, scale_factor=1 / 4, mode='nearest')
        makeup_split_parse_down4_warp=self.warp_basedon_corr(makeup_split_parse_down4,self.transfer_output_data['corr_ref2source'])

        makeup_split_warp_parse_down4 = F.interpolate(self.makeup_split_warp_parse, scale_factor=1 / 4, mode='nearest')
        makeup_split_warp_parse_down4_warp=self.warp_basedon_corr(makeup_split_warp_parse_down4,self.rec_output_data['corr_ref2source'])

        loss_G_sem=self.criterionL1(makeup_split_parse_down4_warp,non_makeup_split_parse_down4)+\
                   self.criterionL1(makeup_split_warp_parse_down4_warp,makeup_split_parse_down4)

        loss_G_sem=loss_G_sem*self.weight_semantic


        transfer_color_face_down4 = F.interpolate(self.transfer_output_data['transfer_img'] * self.non_makeup_face_mask_no_neck_no_ear,
                                                scale_factor=1 / 4, mode='bilinear')
        self.transfer_color_face_down4_warp = self.warp_basedon_corr(transfer_color_face_down4, self.cycle_output_data['corr_ref2source'])
        makeup_color_face_down4 = F.interpolate(self.makeup_color_img * self.makeup_face_mask_no_neck_no_ear,
                                                scale_factor=1 / 4, mode='bilinear')

        self_rec_color_face_down4 = F.interpolate(self.rec_output_data['transfer_img'] * self.makeup_face_mask_no_neck_no_ear,
                                                  scale_factor=1 / 4, mode='bilinear')

        self.self_rec_color_face_down4_warp = self.gen.forward_cross_attention(source_parse=self.makeup_split_warp_parse,
                                                                               ref_parse=self.makeup_split_parse,
                                                                               ref_img=self_rec_color_face_down4)
        makeup_warp_color_face_down4 = F.interpolate(self.makeup_color_change_warp_img * self.makeup_face_warp_mask_no_neck_no_ear,
                                                scale_factor=1 / 4, mode='bilinear')
        loss_G_corr=self.criterionL1(self.transfer_color_face_down4_warp,makeup_color_face_down4)+\
                    self.criterionL1(self.self_rec_color_face_down4_warp,makeup_warp_color_face_down4)

        loss_G_corr=loss_G_corr*self.weight_corr

        # transfer
        # identity
        loss_G_identity = self.criterionIdentity(self.transfer_output_data['transfer_img'], self.non_makeup_color_img)
        loss_G_identity = loss_G_identity * self.weight_identity

        # back
        loss_G_back = self.criterionL1(self.transfer_output_data['transfer_img'] * (1 - self.non_makeup_all_mask[:,0:1,:,:]),
                                       self.non_makeup_color_img * (1 - self.non_makeup_all_mask[:,0:1,:,:]))
        loss_G_back = loss_G_back * self.weight_back

        # Contrastive loss
        positive_distance=self.gram_style(self.transfer_output_data['transfer_img']*self.non_makeup_face_mask_no_neck_no_ear[:,0:1,:,:],
                                      self.makeup_color_img*self.makeup_face_mask_no_neck_no_ear[:,0:1,:,:])
        #print(self.makeup_color_change3_img.shape)
        negative_distance = self.gram_style(self.transfer_output_data['transfer_img'] * self.non_makeup_face_mask_no_neck_no_ear[:, 0:1, :, :],
            self.makeup_color_change_img * self.makeup_face_mask_no_neck_no_ear[:, 0:1, :, :])+\
                            self.gram_style(self.transfer_output_data['transfer_img'] * self.non_makeup_face_mask_no_neck_no_ear[:, 0:1, :, :],
            self.makeup_color_change2_img * self.makeup_face_mask_no_neck_no_ear[:, 0:1, :, :])+\
                            self.gram_style(self.transfer_output_data['transfer_img'] * self.non_makeup_face_mask_no_neck_no_ear[:, 0:1, :, :],
            self.makeup_color_change3_img * self.makeup_face_mask_no_neck_no_ear[:, 0:1, :, :])+\
                            self.gram_style(self.transfer_output_data['transfer_img'] * self.non_makeup_face_mask_no_neck_no_ear[:, 0:1, :, :],
            self.makeup_color_change4_img * self.makeup_face_mask_no_neck_no_ear[:, 0:1, :, :])
        loss_G_contrastive=-torch.log(1.-positive_distance/negative_distance)
        loss_G_contrastive=loss_G_contrastive*self.weight_contrastive

        # adv
        loss_G_GAN = self.backward_G_GAN(self.transfer_output_data['transfer_img'], self.dis)
        loss_G_GAN = loss_G_GAN * self.weight_adv

        # Self supervision
        loss_G_selfL1 = self.criterionL1(self.rec_output_data['transfer_img'], self.makeup_color_change_img)
        loss_G_selfL1 = loss_G_selfL1 * self.weight_self_recL1

        # Cycle
        loss_G_cycleL1 = self.criterionL1(self.cycle_output_data['transfer_img'], self.makeup_color_img)
        loss_G_cycleL1 = loss_G_cycleL1 * self.weight_cycleL1


        loss_G = loss_G_identity + loss_G_GAN + loss_G_selfL1 + loss_G_back + loss_G_sem + loss_G_corr+loss_G_cycleL1+loss_G_contrastive

        loss_G.backward()

        self.G_loss = loss_G.item()
        self.loss_G_identity = loss_G_identity.item()
        self.loss_G_back = loss_G_back.item()
        self.loss_G_GAN = loss_G_GAN.item()
        self.loss_G_selfL1 = loss_G_selfL1.item()
        self.loss_G_cycleL1 = loss_G_cycleL1.item()
        self.loss_G_sem = loss_G_sem.item()
        self.loss_G_contrastive = loss_G_contrastive.item()

    def backward_G_GAN(self, fake, netD):
        outs_fake = netD.forward(fake)
        loss = self.criterionGAN(outs_fake, target_is_real=True, for_discriminator=False)
        return loss

    def set_scheduler(self, opts, last_ep=0):
        self.dis_sch = utils.get_scheduler(self.dis_opt, opts, last_ep)
        self.gen_sch = utils.get_scheduler(self.gen_opt, opts, last_ep)

    def update_lr(self):
        self.dis_sch.step()
        self.gen_sch.step()

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir)
        self.gen.load_state_dict(checkpoint['gen'])
        # optimizer
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
            'gen': self.gen.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return

    def normalize_image(self, x):
        return x[:, 0:3, :, :]

    def assemble_outputs(self):
        # row1
        non_makeup_color_img = self.normalize_image(self.non_makeup_color_img).detach()
        makeup_color_img = self.normalize_image(self.makeup_color_img).detach()
        transfer_content_img=self.normalize_image(self.transfer_output_data['source_face_content']).detach()
        transfer_makeup_img = self.normalize_image(self.transfer_output_data['ref_face_style']).detach()

        transfer_corr1 = F.interpolate(self.makeup_color_img, scale_factor=1 / 4, mode='nearest')
        transfer_corr1=self.gen.forward_cross_attention(source_parse=self.non_makeup_split_parse,
                                                                               ref_parse=self.makeup_split_parse,
                                                                               ref_img=transfer_corr1)
        transfer_corr1 = F.interpolate(transfer_corr1, scale_factor=4, mode='nearest')
        transfer_corr1 = self.normalize_image(transfer_corr1).detach()

        transfer_img = self.normalize_image(self.transfer_output_data['transfer_img']).detach()
        makeup_color_change2_img = self.normalize_image(self.makeup_color_change2_img).detach()

        cycle_content_img = self.normalize_image(self.cycle_output_data['source_face_content']).detach()
        cycle_makeup_img = self.normalize_image(self.cycle_output_data['ref_face_style']).detach()

        cycle_corr1 = F.interpolate(self.transfer_color_face_down4_warp, scale_factor=4, mode='nearest')
        cycle_img=self.normalize_image(self.cycle_output_data['transfer_img'])
        makeup_color_change3_img = self.normalize_image(self.makeup_color_change3_img).detach()

        makeup_color_change_img = self.normalize_image(self.makeup_color_change_img).detach()
        makeup_color_change_warp_img = self.normalize_image(self.makeup_color_change_warp_img).detach()
        rec_content_img = self.normalize_image(self.rec_output_data['source_face_content']).detach()
        rec_makeup_img = self.normalize_image(self.rec_output_data['ref_face_style']).detach()

        rec_corr1 = F.interpolate(self.makeup_color_change_warp_img, scale_factor=1 / 4, mode='nearest')
        rec_corr1 = self.gen.forward_cross_attention(source_parse=self.makeup_split_parse,
                                                          ref_parse=self.makeup_split_warp_parse,
                                                          ref_img=rec_corr1)
        rec_corr1 = F.interpolate(rec_corr1, scale_factor=4, mode='nearest')
        rec_corr1 = self.normalize_image(rec_corr1).detach()
        rec_img = self.normalize_image(self.rec_output_data['transfer_img']).detach()
        makeup_color_change4_img = self.normalize_image(self.makeup_color_change4_img).detach()

        row1 = torch.cat(
            (non_makeup_color_img[0:1, ::], makeup_color_img[0:1, ::],
             transfer_content_img[0:1, ::],transfer_makeup_img[0:1, ::],
             transfer_corr1[0:1, ::], transfer_img[0:1, ::],makeup_color_change2_img[0:1, ::]), 3)

        row2 = torch.cat(
            (makeup_color_img[0:1, ::], transfer_img[0:1, ::],
             cycle_content_img[0:1, ::], cycle_makeup_img[0:1, ::],
             cycle_corr1[0:1, ::],cycle_img[0:1, ::],makeup_color_change3_img[0:1, ::]), 3)

        row3 = torch.cat(
            (makeup_color_change_img[0:1, ::], makeup_color_change_warp_img[0:1, ::],
             rec_content_img[0:1, ::], rec_makeup_img[0:1, ::],
             rec_corr1[0:1, ::],rec_img[0:1, ::],makeup_color_change4_img[0:1, ::]), 3)
        return torch.cat((row1, row2,row3), 2)

    def test_pair(self, data):
        self.non_makeup_color_img = data['non_makeup_color_img'].to(self.gpu).detach()
        self.non_makeup_split_parse = data['non_makeup_split_parse'].to(self.gpu).detach()
        self.non_makeup_all_mask = data['non_makeup_all_mask'].to(self.gpu).detach()
        self.non_makeup_face_mask = data['non_makeup_face_mask'].to(self.gpu).detach()
        self.non_makeup_brow_mask = data['non_makeup_brow_mask'].to(self.gpu).detach()
        self.non_makeup_eye_mask = data['non_makeup_eye_mask'].to(self.gpu).detach()
        self.non_makeup_lip_mask = data['non_makeup_lip_mask'].to(self.gpu).detach()
        self.non_makeup_face_mask_no_neck_no_ear = self.non_makeup_face_mask + self.non_makeup_brow_mask + self.non_makeup_eye_mask + self.non_makeup_lip_mask

        self.makeup_color_img = data['makeup_color_img'].to(self.gpu).detach()
        self.makeup_split_parse = data['makeup_split_parse'].to(self.gpu).detach()
        self.makeup_all_mask = data['makeup_all_mask'].to(self.gpu).detach()
        self.makeup_face_mask = data['makeup_face_mask'].to(self.gpu).detach()
        self.makeup_brow_mask = data['makeup_brow_mask'].to(self.gpu).detach()
        self.makeup_eye_mask = data['makeup_eye_mask'].to(self.gpu).detach()
        self.makeup_lip_mask = data['makeup_lip_mask'].to(self.gpu).detach()
        self.makeup_face_mask_no_neck_no_ear = self.makeup_face_mask + self.makeup_brow_mask + self.makeup_eye_mask + self.makeup_lip_mask

        with torch.no_grad():
            self.transfer_output_data = self.gen(source_img=self.non_makeup_color_img,
                                                 source_parse=self.non_makeup_split_parse,
                                                 source_all_mask=self.non_makeup_all_mask,
                                                 ref_img=self.makeup_color_img,
                                                 ref_parse=self.makeup_split_parse,
                                                 ref_all_mask=self.makeup_all_mask)

        non_makeup_color_img = self.normalize_image(self.non_makeup_color_img).detach()
        makeup_color_img = self.normalize_image(self.makeup_color_img).detach()
        transfer_content_img = self.normalize_image(self.transfer_output_data['source_face_content']).detach()
        transfer_makeup_img = self.normalize_image(self.transfer_output_data['ref_face_style']).detach()

        transfer_corr1 = F.interpolate(self.makeup_color_img, scale_factor=1 / 4, mode='nearest')
        transfer_corr1 = self.gen.forward_cross_attention(source_parse=self.non_makeup_split_parse,
                                                          ref_parse=self.makeup_split_parse,
                                                          ref_img=transfer_corr1)
        non_makeup_all_mask_down4 = F.interpolate(self.non_makeup_all_mask, scale_factor=1 / 4, mode='nearest')
        transfer_corr1 = F.interpolate(transfer_corr1 * non_makeup_all_mask_down4, scale_factor=4, mode='nearest')
        transfer_corr1 = self.normalize_image(transfer_corr1).detach()

        transfer_img = self.normalize_image(self.transfer_output_data['transfer_img']).detach()


        row1 = torch.cat(
            (non_makeup_color_img[0:1, ::], makeup_color_img[0:1, ::],
             transfer_corr1[0:1, ::], transfer_img[0:1, ::]),
            3)

        return row1

   