import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from CSD_MT.utils import init_net
from CSD_MT.modules import Generator


class CSD_MT(nn.Module):
    def __init__(self, opts):
        super(CSD_MT, self).__init__()
        # parameters
        self.opts = opts
        
        
        self.batch_size = opts.batch_size
        # self.gpu = torch.device('cuda:{}'.format(opts.gpu)) if torch.cuda.is_available() else torch.device('cpu')
        self.gpu=torch.device('cpu')
        self._build_model()

    def _build_model(self):
        print('start build model')
        self.gen = init_net(Generator(input_dim=3,parse_dim=self.opts.semantic_dim,ngf=16,device=self.gpu), self.gpu,init_type='normal', gain=0.02)
        print('finish build model')

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir,map_location=torch.device('cpu'))
        self.gen.load_state_dict(checkpoint['gen'])
        # optimizer
        return checkpoint['ep'], checkpoint['total_it']

    def normalize_image(self, x):
        return x[:, 0:3, :, :]


    def test_pair(self, data):
        self.non_makeup_color_img = data['non_makeup_color_img'].to(self.gpu).detach()
        self.non_makeup_split_parse = data['non_makeup_split_parse'].to(self.gpu).detach()
        self.non_makeup_all_mask = data['non_makeup_all_mask'].to(self.gpu).detach()


        self.makeup_color_img = data['makeup_color_img'].to(self.gpu).detach()
        self.makeup_split_parse = data['makeup_split_parse'].to(self.gpu).detach()
        self.makeup_all_mask = data['makeup_all_mask'].to(self.gpu).detach()


        with torch.no_grad():
            self.transfer_output_data = self.gen(source_img=self.non_makeup_color_img,
                                                 source_parse=self.non_makeup_split_parse,
                                                 source_all_mask=self.non_makeup_all_mask,
                                                 ref_img=self.makeup_color_img,
                                                 ref_parse=self.makeup_split_parse,
                                                 ref_all_mask=self.makeup_all_mask)


        transfer_img = self.normalize_image(self.transfer_output_data['transfer_img']).detach()
        return transfer_img[0:1, ::]

    