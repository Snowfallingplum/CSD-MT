import torch
import random
import functools
import torch.nn as nn
import torch.nn.functional as F

####################################################################
# ------------------------- Generator --------------------------
####################################################################

class Generator(nn.Module):
    def __init__(self,input_dim,parse_dim,ngf,device):
        super(Generator, self).__init__()
        self.ngf=ngf
        self.content_style_separation = Content_Style_Separation(device)
        self.semantic_enc=Encoder_Semantic(parse_dim=parse_dim+2, ngf=ngf)
        self.content_enc=Encoder_down4(input_dim=input_dim+1, ngf=ngf*2)
        self.makeup_enc = Encoder_down4(input_dim=input_dim, ngf=ngf*2)
        self.cross_atten=Attention(channels=ngf*4, norm='Instance', sn=False)
        self.dec=Decoder_up4(ngf=ngf*2)

    def forward_cross_attention(self,source_parse,ref_parse,ref_img):
        source_semantic_f = self.semantic_enc(source_parse)
        ref_semantic_f = self.semantic_enc(ref_parse)
        ref_warp_img, corr_ref2source = self.cross_atten(source_semantic_f, ref_semantic_f, ref_img)
        return ref_warp_img

    def forward(self,source_img,source_parse,source_all_mask,ref_img,ref_parse,ref_all_mask):
        source_back = source_img * (1. - source_all_mask)

        source_face = source_img * source_all_mask
        source_face_content, source_face_style = self.content_style_separation(source_face)

        ref_face = ref_img * ref_all_mask
        ref_face_content, ref_face_style = self.content_style_separation(ref_face)

        source_semantic_f=self.semantic_enc(source_parse)
        ref_semantic_f=self.semantic_enc(ref_parse)

        source_content_f_list=self.content_enc(torch.cat([source_face_content,source_back],dim=1))
        
        source_makeup = F.interpolate(source_face_style, scale_factor=0.25, mode='bilinear')
        ref_makeup=F.interpolate(ref_face_style,scale_factor=0.25,mode='bilinear')
        ref_makeup_warp,corr_ref2source=self.cross_atten(source_semantic_f,ref_semantic_f,ref_makeup)

        ref_makeup_warp=ref_makeup_warp*F.interpolate(source_all_mask,scale_factor=0.25)

        transfer_img=self.dec(source_content_f_list,ref_makeup_warp)
        output_data={'source_face_content':source_face_content,'ref_face_style':ref_face_style,
                     'transfer_img':transfer_img,'corr_ref2source':corr_ref2source,
                     'ref_makeup_warp':ref_makeup_warp}
        return output_data



class Content_Style_Separation(nn.Module):
    def __init__(self,device):
        super(Content_Style_Separation, self).__init__()
        self.kernel = self.gauss_kernel(device)
        self.device=device

    def gauss_kernel(self, device, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel
    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def downsample(self, x):
        return x[:, :, ::2, ::2]  # down-sampling

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def forward(self,img):
        current = img
        filtered = self.conv_gauss(current, self.kernel)
        down = self.downsample(filtered)
        up = self.upsample(down)
        content=current - up
        style=up
        # Grayscale
        content = 0.299 * content[:, 0:1, ::] + 0.587 * content[:, 1:2,::] + 0.114 * content[:,2:3, ::]
        content=content*(0.6+0.6*random.random())
        return content, style


####################################################################
# ------------------------- Encoder --------------------------
####################################################################

class Encoder_Semantic(nn.Module):
    def __init__(self, parse_dim, ngf=32):
        super(Encoder_Semantic, self).__init__()
        self.parse_dim = parse_dim
        self.conv1 = LeakyReLUConv2d(parse_dim, ngf * 1, kernel_size=7, stride=1, padding=3, norm='instance')
        self.conv2 = LeakyReLUConv2d(ngf * 1, ngf * 2, kernel_size=3, stride=2, padding=1, norm='instance')
        self.conv3 = LeakyReLUConv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, norm='instance')
        self.conv4 = LeakyReLUConv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, norm='instance')
        self.conv5 = LeakyReLUConv2d(ngf * 8, ngf * 16, kernel_size=3, stride=2, padding=1, norm='instance')
        self.res1 = ResBlock(channels=ngf * 16)
        self.res2 = ResBlock(channels=ngf * 16)
        self.up1 = Upsample(in_channels=ngf * 16, out_channels=ngf * 8, is_up=True)
        self.up2 = Upsample(in_channels=ngf * 8, out_channels=ngf * 4, is_up=True)

    def forward(self, parse):
        ins_feat = parse  # 当前实例特征tensor
        # 生成从-1到1的线性值
        x_range = torch.linspace(-1, 1, ins_feat.shape[-1], device=ins_feat.device)
        y_range = torch.linspace(-1, 1, ins_feat.shape[-2], device=ins_feat.device)
        y, x = torch.meshgrid(y_range, x_range)  # 生成二维坐标网格
        y = y.expand([ins_feat.shape[0], 1, -1, -1])  # 扩充到和ins_feat相同维度
        x = x.expand([ins_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)  # 位置特征
        input = torch.cat([ins_feat, coord_feat], 1)  # concatnate一起作为下一个卷积的输入
        output1 = self.conv1(input)
        output2 = self.conv2(output1)
        output3 = self.conv3(output2)
        output4 = self.conv4(output3)
        output5 = self.conv5(output4)
        output = self.res1(output5)
        output = self.res2(output)
        output = self.up1(output+output5)
        output = self.up2(output + output4)
        return output


class Encoder_down4(nn.Module):
    def __init__(self, input_dim, ngf=64):
        super(Encoder_down4, self).__init__()
        # identity encoder
        self.conv1 = LeakyReLUConv2d(input_dim, ngf * 1, kernel_size=3, stride=1, padding=1, norm='instance')
        self.conv2 = LeakyReLUConv2d(ngf * 1, ngf * 2, kernel_size=3, stride=2, padding=1, norm='instance')
        self.conv3 = LeakyReLUConv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, norm='instance')
        self.res1 = ResBlock(channels=ngf * 4)
        self.res2 = ResBlock(channels=ngf * 4)

    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.res1(out_3)
        out_4 = self.res2(out_4)
        out_list = [out_1, out_2, out_3, out_4]
        return out_list


####################################################################
# ------------------------- Attention -----------------------------
####################################################################
class Attention(nn.Module):
    def __init__(self, channels, norm='Instance', sn=False):
        super(Attention, self).__init__()
        in_dim = channels
        self.chanel_in = in_dim
        self.softmax_alpha = 100
        self.eps = 1e-5
        self.fa_conv = LeakyReLUConv2d(in_dim, in_dim // 4, kernel_size=1, stride=1, padding=0, norm=norm, sn=sn)
        self.fb_conv = LeakyReLUConv2d(in_dim, in_dim // 4, kernel_size=1, stride=1, padding=0, norm=norm, sn=sn)

    def cal_correlation(self, fa, fb, alpha):
        '''
            calculate correspondence matrix and warp the exemplar features
        '''
        assert fa.shape == fb.shape, \
            'Feature shape must match. Got %s in a and %s in b)' % (fa.shape, fb.shape)
        n, c, h, w = fa.shape
        # subtract mean
        fa = fa - torch.mean(fa, dim=(2, 3), keepdim=True)
        fb = fb - torch.mean(fb, dim=(2, 3), keepdim=True)

        # vectorize (merge dim H, W) and normalize channelwise vectors
        fa = fa.view(n, c, -1)
        fb = fb.view(n, c, -1)
        fa = fa / (torch.norm(fa, dim=1, keepdim=True) + self.eps)
        fb = fb / (torch.norm(fb, dim=1, keepdim=True) + self.eps)

        energy_ab_T = torch.bmm(fb.transpose(-2, -1), fa) * alpha
        corr_ab_T = F.softmax(energy_ab_T, dim=1)  # n*HW*C @ n*C*HW -> n*HW*HW
        return corr_ab_T

    def forward(self, fa_raw, fb_raw, fc_raw):
        fa = self.fa_conv(fa_raw)
        fb = self.fb_conv(fb_raw)
        corr_ab_T = self.cal_correlation(fa, fb, self.softmax_alpha)
        n, c, h, w = fc_raw.shape
        fc_raw_warp = torch.bmm(fc_raw.view(n, c, h * w), corr_ab_T)  # n*HW*1
        fc_raw_warp = fc_raw_warp.view(n, c, h, w)
        return fc_raw_warp, corr_ab_T


####################################################################
# ------------------------- Decoder --------------------------
####################################################################

class Decoder_up4(nn.Module):
    def __init__(self, ngf=64):
        super(Decoder_up4, self).__init__()
        self.spade1 = SPADEResnetBlock(fin=ngf * 4, fout=ngf * 4, semantic_nc=3)
        self.spade2 = SPADEResnetBlock(fin=ngf * 4, fout=ngf * 4, semantic_nc=3)
        # self.spade3 = SPADEResnetBlock(fin=ngf * 4, fout=ngf * 4, semantic_nc=3)
        # self.spade4 = SPADEResnetBlock(fin=ngf * 4, fout=ngf * 4, semantic_nc=3)
        self.up1 = Upsample(in_channels=ngf * 4, out_channels=ngf * 2, is_up=True)
        self.spade5 = SPADEResnetBlock(fin=ngf * 2, fout=ngf * 2, semantic_nc=3)
        self.up2 = Upsample(in_channels=ngf * 2, out_channels=ngf * 1, is_up=True)
        self.spade6 = SPADEResnetBlock(fin=ngf * 1, fout=ngf * 1, semantic_nc=3)
        self.last_conv = LeakyReLUConv2d(ngf * 1, ngf * 1, kernel_size=3, stride=1, padding=1, norm='instance')
        self.img_conv = nn.Conv2d(ngf * 1, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, content_list, makeup):
        y = content_list[-1]
        y = self.spade1(y,makeup)
        y = self.spade2(y, makeup)
        # y = self.spade3(y, makeup)
        # y = self.spade4(y, makeup)
        y = self.up1(y + content_list[-2])
        y = self.spade5(y, makeup)
        y = self.up2(y + content_list[-3])
        y = self.spade6(y, makeup)
        y = self.last_conv(y + content_list[-4])
        y = self.img_conv(y)
        y = self.tanh(y)
        return y


####################################################################
# ---------------------------- Blocks -----------------------------
####################################################################

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, semantic_nc):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # define normalization layers
        self.norm_0 = SPADE(fin, semantic_nc)
        self.norm_1 = SPADE(fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        ks = 3
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class Upsample(nn.Module):
    """upsample Block with conditional instance normalization."""

    def __init__(self, in_channels, out_channels, is_up=True):
        super(Upsample, self).__init__()
        self.is_up = is_up
        if self.is_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.actv = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        if self.is_up:
            x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.actv(x)
        return x


class ResBlock(nn.Module):
    """Residual Block with conditional instance normalization."""

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.nn1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.nn1(y)
        y = self.conv2(y)
        y = self.norm2(y)
        return x + y


class LeakyReLUConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [nn.utils.spectral_norm(
                nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(outplanes, affine=False)]
        model += [nn.LeakyReLU(0.2, inplace=False)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
