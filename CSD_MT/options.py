import argparse
class Options():
    def __init__(self):
        self.parser=argparse.ArgumentParser()
        # data loader related
        self.parser.add_argument('--phase', type=str, default='test', help='phase for train or test')
        self.parser.add_argument('--input_dim', type=int, default=3, help='input_dim')
        
        self.parser.add_argument('--output_dim', type=int, default=3, help='output_dim')
        self.parser.add_argument('--semantic_dim', type=int, default=10, help='semantic_dim')

        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
        self.parser.add_argument('--crop_size', type=int, default=256, help='cropped image size for training')
       
        self.parser.add_argument('--nThreads', type=int, default=0, help='# of threads for data loader')

        # log related
        self.parser.add_argument('--model_dir', type=str, default='./weights', help='path for saving result models')
        self.parser.add_argument('--init_type', type=str, default='normal', help='init_type [normal, xavier, kaiming, orthogonal]')
        self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt