# -*- coding: utf-8 -*-
"""CModie.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mI5euyw3M1qQpJv0QqBRYjNM6ZMwFj2M
"""

import argparse
import json
import os
import re
from pathlib import Path
from sconf import Config
import numpy as np
import torch
from datasets import load_dataset, config
from PIL import Image
from tqdm import tqdm
import time
from donut import DonutModel, JSONParseEvaluator, load_json, save_json
from transformers import DonutProcessor, VisionEncoderDecoderModel
from lightning_module import DonutDataPLModule, DonutModelPLModule
from utils.util import prepare_input
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib import cm
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.util import prepare_input

device = "cuda"

checkpoint = "naver-clova-ix/donut-base"

processor = DonutProcessor.from_pretrained(checkpoint)
# pretrained_model = VisionEncoderDecoderModel.from_pretrained(checkpoint)

config = Config("config/train_cord.yaml")

class DonutEncoderModel(nn.Module):
    def __init__(self, checkpoint="naver-clova-ix/donut-base"):
        #self.processor = DonutProcessor.from_pretrained(checkpoint)
        super(DonutEncoderModel, self).__init__()
        self.pretrained_model_clear = VisionEncoderDecoderModel.from_pretrained(checkpoint)
        self.pretrained_model_blur = VisionEncoderDecoderModel.from_pretrained(checkpoint)
        self.pretrained_model_clear.requires_grad = False
        self.pretrained_model_blur.requires_grad = True

    def forward(self, clear_image_tensors, blur_image_tensors):
        # Add your model inference logic here
        clear_embedding = self.pretrained_model_clear.encoder(clear_image_tensors).last_hidden_state.squeeze(0)
        blur_embedding = self.pretrained_model_blur.encoder(blur_image_tensors).last_hidden_state.squeeze(0)
        return clear_embedding, blur_embedding

def custom_collate_fn(batch):

    clear_images = [prepare_input(config, img['image']) for img in batch]
    blur_images = [prepare_input(config, img['image'], add_blur=True) for img in batch]

    clear_images = torch.stack(clear_images)
    blur_images = torch.stack(blur_images)
    return {"clear_images" : clear_images, "blur_images" : blur_images}

dataset = load_dataset("naver-clova-ix/cord-v2",batch_size=1, split="train")

train_loader = DataLoader(dataset, batch_size=1,  collate_fn=custom_collate_fn, shuffle=True)

model = DonutEncoderModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CosineEmbeddingLoss()

model.to(device)

# def viz(args):
#     checkpoint = "naver-clova-ix/donut-base"

#     # processor = DonutProcessor.from_pretrained(checkpoint)
#     # pretrained_model = VisionEncoderDecoderModel.from_pretrained(checkpoint)

#     # pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)

#     if torch.cuda.is_available():
#         pretrained_model.half()
#         pretrained_model.to("cuda")

#     # pretrained_model.eval()
#     dataset = load_dataset(args.dataset_name_or_paths[0], split="validation")
#     embeds_normal, embeds_blur = [], []
#     print("Loaded model and dataset!!")
#     i = 0

#     tsne = TSNE(2, verbose=1)
#     cmap = cm.get_cmap('tab10')
#     fig, ax = plt.subplots(figsize=(8,8))
#     embeds, embeds_blur = [], []
#     for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
#         image_tensors = prepare_input(config, sample["image"]).unsqueeze(0)

#         if torch.cuda.is_available():  # half is not compatible in cpu implementation.
#             # image_tensors = image_tensors.half()
#             image_tensors = image_tensors.to("cuda")

#         # output = pretrained_model.encoder(image_tensors).last_hidden_state.squeeze(0)
#         # print(output.shape)
#         # embeds_normal.append(output)

#         image_tensors_blur = prepare_input(config, sample["image"], add_blur=True).unsqueeze(0)

#         if torch.cuda.is_available():  # half is not compatible in cpu implementation.
#             # image_tensors_blur = image_tensors_blur.half()
#             image_tensors_blur = image_tensors_blur.to("cuda")

#         output, output_blur = model(image_tensors, image_tensors_blur)
#         # output_blur = pretrained_model.encoder(image_tensors_blur).last_hidden_state.squeeze(0)
#         print(output_blur.shape, output.shape)
#         # embeds_blur.append(output_blur)
#         # print("Generated :: ", idx)
#         # import pdb;pdb.set_trace()

#         print("Plotting embeddings!!")
#         # embeds_normal, embeds_blur = np.array(embeds_normal), np.array(embeds_blur)
#         # import pdb;pdb.set_trace();
#         embeds.append(output.to('cpu').detach().numpy())
#         embeds_blur.append(output_blur.to('cpu').detach().numpy())

#         del image_tensors_blur
#         del image_tensors
#         del output
#         del output_blur

#         torch.cuda.empty_cache()
#         time.sleep(0.2)     # giving some time to clear!!
#         i += 1
#         if i == 1:
#             break

#     for i in range(len(embeds)):
#         fig, ax = plt.subplots(figsize=(8,8))
#         proj_normal = tsne.fit_transform(np.array(embeds[i]))
#         proj_blur = tsne.fit_transform(np.array(embeds_blur[i]))
#         # proj_normal = tsne.fit_transform(embeds_normal)
#         # proj_blur = tsne.fit_transform(embeds_blur)

#         ax.scatter(proj_normal[:,0],proj_normal[:,1], c=np.array(cmap(0)).reshape(1,4), label = "Normal" ,alpha=0.5)
#         ax.scatter(proj_blur[:,0],proj_blur[:,1], c=np.array(cmap(1)).reshape(1,4), label = "With Blur" ,alpha=0.5)

#         ax.legend(fontsize='large', markerscale=2)
#         plt.savefig(f'modie_embeddings_trained_{i}.png')

for epoch in range(100):
    model.train()
    for batch in train_loader:
        #print(batch['clear_images'], torch.flatten(clear_img_tensor,start_dim=1).shape)
        clear_img_tensor  = batch['clear_images'].to(device)
        blur_img_tensor  = batch['blur_images'].to(device)
        optimizer.zero_grad()
        torch_zeros = torch.ones(clear_img_tensor.shape[0]).to(device)

        clear_img_embed, blur_img_embed = model(clear_img_tensor, blur_img_tensor)

        loss = criterion(torch.flatten(clear_img_embed,start_dim=1), torch.flatten(blur_img_embed,start_dim=1), torch_zeros)
        print(epoch, loss)
        # viz(config)
        loss.backward()
        optimizer.step()












