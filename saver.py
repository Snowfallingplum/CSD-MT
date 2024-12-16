import os
import cv2
import numpy as np
import torchvision
from tensorboardX import SummaryWriter


class Saver():
    def __init__(self, opts):
        self.log_dir = opts.log_dir
        self.image_dir = opts.img_dir
        self.model_dir = opts.model_dir

        self.log_freq = opts.log_freq
        self.img_save_freq = opts.img_save_freq
        self.model_save_freq = opts.model_save_freq

        # create tensorboard writer
        self.writer = SummaryWriter(logdir=self.log_dir)

    # write losses and images to tensorboard
    def write_log(self, total_it, model):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if (total_it + 1) % self.log_freq == 0:
            # write loss
            members = [attr for attr in dir(model) if
                       not callable(getattr(model, attr)) and not attr.startswith("__") and 'loss' in attr]
            for m in members:
                self.writer.add_scalar(m, getattr(model, m), total_it)

    # save result images
    def write_img(self, ep, model):
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
        if (ep + 1) % self.img_save_freq == 0:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/self_%05d.jpg' % (self.image_dir, ep)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)
        elif ep == -1:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/self_last.jpg' % (self.image_dir)
            torchvision.utils.save_image(assembled_images / 2 + 0.5, img_filename, nrow=1)

    # save model
    def write_model(self, ep, total_it, model):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if (ep + 1) % self.model_save_freq == 0:
            print('--- save the model @ ep %d ---' % (ep))
            model.save('%s/makeup_%05d.pth' % (self.model_dir, ep), ep, total_it)
        elif ep == -1:
            model.save('%s/makeup_last.pth' % self.model_dir, ep, total_it)

    # save test pair images
    def write_test_pair_img(self, iter, model, data):
        root = os.path.join(self.image_dir, 'test_pair')
        if not os.path.exists(root):
            os.makedirs(root)
        import time
        start_time=time.time()
        test_pair_img = model.test_pair(data)
        print(time.time()-start_time)
        img_filename = '%s/gen_%05d.jpg' % (root, iter)
        torchvision.utils.save_image(test_pair_img / 2 + 0.5, img_filename, nrow=1)

   