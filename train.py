import os
import torch
from saver import Saver
from options import Options
from model import CSDMT
from dataset import MakeupDataset

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def train():
    # parse options
    parser = Options()
    opts = parser.parse()

    # load dataset
    print('\n--- load dataset ---')
    dataset = MakeupDataset(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True,
                                               num_workers=opts.nThreads)
    print('dataset size:',len(dataset))
    # load model
    print('\n--- load model ---')
    model = CSDMT(opts)
    if opts.resume is None:
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d' % (ep0))
    # saver for  output
    saver = Saver(opts)
    # train
    print('\n--- train ---')
    max_it = 1000000
    for ep in range(ep0, opts.n_ep):
        for it, data in enumerate(train_loader):
            # load data to gpu
            model.load_data(data)
            # updata corr network
            model.update_D()
            model.update_G()
            # print
            if it%10==0:
                print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.dis_opt.param_groups[0]['lr']))
                print('G_loss: %.2f,loss_G_contrastive: %.2f,loss_G_GAN: %.2f,loss_G_selfL1: %.2f,loss_G_cycleL1: %.2f ' %
                      (model.G_loss,model.loss_G_contrastive,model.loss_G_GAN,model.loss_G_selfL1,model.loss_G_cycleL1 ))
            total_it += 1
            # save log
            saver.write_log(total_it, model)
            # save the last model
            if total_it >= max_it:
                saver.write_img(-1, model)
                saver.write_model(-1, total_it,model)
                break
        # decay learning rate
        if opts.n_ep_decay > -1:
            model.update_lr()
        # save result image
        saver.write_img(ep, model)
        # Save network weights
        saver.write_model(ep, total_it, model)


if __name__ == '__main__':
    train()
    print('The training is complete')
