import os
import torch
from saver import Saver
from options import Options
from model import CSDMT
from dataset import MakeupDataset
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def eval_pair():
    # parse options
    parser = Options()
    opts = parser.parse()
    opts.phase = 'test_pair'
    opts.data_root='./examples/images'
    opts.img_dir='./inference_results'
    # data loader
    print('\n--- load dataset ---')
    dataset = MakeupDataset(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,num_workers=0)

    # model
    print('\n--- load model ---')
    model = CSDMT(opts)
    ep0, total_it = model.resume(os.path.join(opts.model_dir, 'makeup_00499.pth'))
    model.eval()
    print('start test pair')
    # saver for display and output
    saver = Saver(opts)
    for iter, data in enumerate(train_loader):
        with torch.no_grad():
            saver.write_test_pair_img(iter, model,data)

if __name__=='__main__':
    eval_pair()
   
