import os
import torch
import cv2
import time
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr
import CSD_MT_eval

def get_makeup_transfer_results256(non_makeup_img,makeup_img):
    transfer_img=CSD_MT_eval.makeup_transfer256(non_makeup_img,makeup_img)
    return transfer_img


example = {}
non_makeup_dir = 'examples/non_makeup'
makeup_dir = 'examples/makeup'
non_makeup_list = [os.path.join(non_makeup_dir, file) for file in os.listdir(non_makeup_dir)]
non_makeup_list.sort()
makeup_list = [os.path.join(makeup_dir, file) for file in os.listdir(makeup_dir)]
makeup_list.sort()

with gr.Blocks() as demo:
    with gr.Group():
        with gr.Tab("CSD-MT"):
            with gr.Row():
                with gr.Column():
                    non_makeup = gr.Image(source='upload', elem_id="image_upload", type="numpy",
                                          label="Non_makeup Image")
                    gr.Examples(non_makeup_list, inputs=[non_makeup], label="Examples - Non_makeup Image",
                                examples_per_page=6)

                    makeup = gr.Image(source='upload', elem_id="image_upload", type="numpy",
                                      label="Makeup Image")
                    gr.Examples(makeup_list, inputs=[makeup], label="Examples - Makeup Image", examples_per_page=6)

                with gr.Column():
                    image_out = gr.Image(label="Output",  type="numpy", elem_id="output-img").style(height=256)

                    with gr.Row().style(mobile_collapse=False, equal_height=True):
                        btn = gr.Button("Makeup Transfer! (CSD-MT)").style()
            btn.click(fn=get_makeup_transfer_results256, inputs=[non_makeup,makeup],
                      outputs=image_out)


demo.launch()
