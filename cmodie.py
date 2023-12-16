# -*- coding: utf-8 -*-


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
from embedding_viz import viz_modie
from utils.loss import CustomLoss, AntiContrastiveLoss, MMD
import torch.nn.functional as F

device = "cuda"

checkpoint = "naver-clova-ix/donut-base"
config = Config("config/train_cord.yaml")

class DonutEncoderModel(nn.Module):
    def __init__(self, checkpoint="naver-clova-ix/donut-base"):
        super(DonutEncoderModel, self).__init__()
        self.t_normal =  VisionEncoderDecoderModel.from_pretrained(checkpoint)
        self.t_blur =  VisionEncoderDecoderModel.from_pretrained(checkpoint)

        self.normal_model = self.t_normal.encoder
        self.blur_model = self.t_blur.encoder

        self.normal_model.requires_grad = False
        self.blur_model.requires_grad = True

        del self.t_normal
        del self.t_blur

    def forward(self, clear_image_tensors, blur_image_tensors):
        clear_embedding = self.normal_model(clear_image_tensors).last_hidden_state.squeeze(0)
        blur_embedding = self.blur_model(blur_image_tensors).last_hidden_state.squeeze(0)
        return clear_embedding, blur_embedding

def custom_collate_fn(batch):
    config.add_blur = False
    clear_images = [prepare_input(config, img['image'], random_padding = True) for img in batch]
    config.add_blur = True
    blur_images = [prepare_input(config, img['image'], random_padding = True) for img in batch]

    clear_images = torch.stack(clear_images)
    blur_images = torch.stack(blur_images)
    return {"clear_images" : clear_images, "blur_images" : blur_images}

# You might need to experiment with these values
sigma_value = 0.8  # Try different values


torch.cuda.empty_cache()
time.sleep(5)     # giving some time to clear!!

dataset = load_dataset("naver-clova-ix/cord-v2")
train_loader = DataLoader(dataset['train'], batch_size=2,  collate_fn=custom_collate_fn, shuffle=True)

model = DonutEncoderModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
# criterion = nn.CosineEmbeddingLoss()
# criterion = nn.MSELoss()
# criterion = AntiContrastiveLoss(margin=0)
# criterion = CustomLoss(weight_mean_loss=0.5, weight_distance_loss=0.05, weight_dispersion_loss = 0.45)

model.to(device)
def save(model, epoch, loss):
  torch.save({
        'epoch': epoch,
        'normal_model_state_dict': model.normal_model.state_dict(),
        'blur_model_state_dict': model.blur_model.state_dict(),
        'loss': loss
    }, f'model-cmodie-epoch-{epoch}.pth')

# freezing the normal model encoder
for name, param in model.named_parameters():
  if 'normal_model' in name:
    param.requires_grad = False
  # else:
  #   print(f"Layer {name} has requires_grad=True.")


model.eval()
with torch.no_grad():  # Disable gradient computation
  viz_modie(config, model, dataset['validation'], plot_idx = "pre")

import time
for epoch in range(10):
    model.train()
    save(model, 0, 1)
    st= time.time()
    for iter, batch in enumerate(train_loader):
        clear_img_tensor = batch['clear_images'].to(device)
        blur_img_tensor = batch['blur_images'].to(device)
        optimizer.zero_grad()
        # torch_ones = torch.ones(clear_img_tensor.shape[0]).to(device)  # required for cosine embedding loss!
        clear_img_embed, blur_img_embed = model(clear_img_tensor, blur_img_tensor)
        normal_embeds = torch.flatten(clear_img_embed, start_dim=1)
        blur_embeds = torch.flatten(blur_img_embed, start_dim=1)
        # loss = criterion(torch.nn.functional.normalize(normal_embeds, p=2, dim=1),
        #                  torch.nn.functional.normalize(blur_embeds, p=2, dim=1))
        # loss = criterion(torch.nn.functional.normalize(normal_embeds, p=2, dim=1),
        #                  torch.nn.functional.normalize(blur_embeds, p=2, dim=1), torch_ones)
        # loss = criterion(normal_embeds, blur_embeds, torch_ones)
        # loss = criterion(normal_embeds, blur_embeds)
        # loss = criterion(blur_embeds, normal_embeds)
        loss = MMD(blur_embeds, normal_embeds, kernel = "rbf")
        loss.backward()
        optimizer.step()
        
        if iter % 10 == 0:  # Print every 10 iterations
            print(f'Epoch {epoch}, Iteration {iter}, Loss {loss.item()}')

    if epoch and (epoch % 2 == 0):
      # Reduce the learning rate for every 2 epochs
      scheduler.step()
    print(f"Epoch {epoch} has lr -> {scheduler.get_last_lr()} time -> {time.time() - st} secs")
    model.eval()
    with torch.no_grad():  # Disable gradient computation
      viz_modie(config, model, dataset['validation'], plot_idx = "after_"+str(epoch + 1))
    save(model, epoch + 1, loss)

viz_modie(config, model, dataset['validation'], plot_idx = "final_"+str(epoch))
