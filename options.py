import argparse
class Options():
    def __init__(self):
        self.parser=argparse.ArgumentParser()
        # data loader related
        self.parser.add_argument('--data_root', type=str, default='../MT-Dataset/images', help='path of data')
        self.parser.add_argument('--phase', type=str, default='train', help='phase for train or test')
        self.parser.add_argument('--input_dim', type=int, default=3, help='input_dim')
        self.parser.add_argument('--output_dim', type=int, default=3, help='output_dim')
        self.parser.add_argument('--semantic_dim', type=int, default=10, help='semantic_dim')

        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--resize_size', type=int, default=286, help='resized image size for training')
        self.parser.add_argument('--crop_size', type=int, default=256, help='cropped image size for training')
        self.parser.add_argument('--flip', type=bool, default=True, help='specified if  flipping')
        self.parser.add_argument('--nThreads', type=int, default=4, help='# of threads for data loader')

        # log related
        self.parser.add_argument('--name', type=str, default='CSD-MT', help='folder name to save outputs')
        self.parser.add_argument('--log_dir', type=str, default='./logs', help='path for saving display results')
        self.parser.add_argument('--img_dir', type=str, default='./imgs', help='path for saving result images')
        self.parser.add_argument('--model_dir', type=str, default='./weights', help='path for saving result models')
        self.parser.add_argument('--log_freq', type=int, default=10, help='freq (iteration) of log')
        self.parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
        self.parser.add_argument('--model_save_freq', type=int, default=50, help='freq (epoch) of saving models')

        # weight
        self.parser.add_argument('--weight_self_recL1', type=float, default=10, help='weight_self_recL1')
        self.parser.add_argument('--weight_cycleL1', type=float, default=10, help='weight_cycleL1')
        self.parser.add_argument('--weight_back', type=float, default=5, help='weight_back')
        self.parser.add_argument('--weight_contrastive', type=float, default=1, help='weight_contrastive')
        self.parser.add_argument('--weight_identity', type=float, default=0.1, help='weight_identity')
        self.parser.add_argument('--weight_semantic', type=float, default=0, help='weight_semantic')
        self.parser.add_argument('--weight_adv', type=float, default=2, help='weight_adv')
        self.parser.add_argument('--weight_corr', type=float, default=1, help='weight_corr')


        # training related
        # discriminator
        self.parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
        self.parser.add_argument('--dis_norm', type=str, default='None',
                                 help='normalization layer in discriminator [None, Instance]')
        self.parser.add_argument('--dis_sn', type=bool, default=True,
                                 help='use spectral normalization in discriminator')
        # generator
        self.parser.add_argument('--G_sn', type=bool,default=True, help='use spectral normalization in generator')
        self.parser.add_argument('--gan_mode', type=str, default='original', help='gan_mode')

        self.parser.add_argument('--gpu', type=int, default=0, help='gpu: e.g. 0 ,use -1 for CPU')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate of discriminator')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
        self.parser.add_argument('--init_type', type=str, default='normal', help='init_type [normal, xavier, kaiming, orthogonal]')
        self.parser.add_argument('--n_ep', type=int, default=500, help='number of epochs')
        self.parser.add_argument('--n_ep_decay', type=int, default=250, help='epoch start decay learning rate, set -1 if no decay')
        self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt