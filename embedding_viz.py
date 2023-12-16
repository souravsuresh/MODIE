"""
MODIE Embedding VIZ
"""
import argparse
import json
import os
import re
from pathlib import Path
from sconf import Config
import numpy as np
import torch
from datasets import load_dataset
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
import torch.nn.functional as F 


def plot_embeds(embeds, embeds_blur, plot_idx = 0):
    tsne = TSNE(2, verbose=1)
    cmap = cm.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(8,8))
    for i in range(len(embeds)):
        fig, ax = plt.subplots(figsize=(8,8))
        proj_normal = tsne.fit_transform(np.array(embeds[i]))
        proj_blur = tsne.fit_transform(np.array(embeds_blur[i]))

        ax.scatter(proj_normal[:,0],proj_normal[:,1], c=np.array(cmap(0)).reshape(1,4), label = "Normal" ,alpha=0.5)
        ax.scatter(proj_blur[:,0],proj_blur[:,1], c=np.array(cmap(1)).reshape(1,4), label = "With Blur" ,alpha=0.5)

        ax.legend(fontsize='large', markerscale=2)
        plt.savefig(f'modie_embeddings_initial_{plot_idx}_{i}.png')

def viz_modie(config, model, dataset, n_plots = 1, plot_idx = 0):

    torch.cuda.empty_cache()
    time.sleep(1)     # giving some time to clear!!
    embeds, embeds_blur = [], []
    
    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        config.add_blur = False
        image_tensors = prepare_input(config, sample["image"]).unsqueeze(0)        
        if torch.cuda.is_available(): 
            image_tensors = image_tensors.to("cuda")
        config.add_blur = True
        image_tensors_blur = prepare_input(config, sample["image"]).unsqueeze(0)
        if torch.cuda.is_available():
            image_tensors_blur = image_tensors_blur.to("cuda")

        output = torch.nn.functional.normalize(model.normal_model(image_tensors).last_hidden_state.squeeze(0), p=2, dim=1)
        output_blur = torch.nn.functional.normalize(model.blur_model(image_tensors_blur).last_hidden_state.squeeze(0), p=2, dim=1)
        
        embeds.append(output.to('cpu').detach().numpy())
        embeds_blur.append(output_blur.to('cpu').detach().numpy())

        del image_tensors_blur
        del image_tensors
        del output
        del output_blur

        torch.cuda.empty_cache()
        time.sleep(0.2)     # giving some time to clear!!
        print(f"Done Plotting embeddings for {idx}!!")
        n_plots -= 1
        if n_plots == 0:
            break

    plot_embeds(embeds, embeds_blur, plot_idx)


def viz(args, pretrained_model = None):
    checkpoint = "naver-clova-ix/donut-base"
    if pretrained_model == None:
        pretrained_model = VisionEncoderDecoderModel.from_pretrained(checkpoint)

        if torch.cuda.is_available():
            pretrained_model.half()
            pretrained_model.to("cuda")

        pretrained_model.eval()

    dataset = load_dataset(args.dataset_name_or_paths[0], split="validation")
    embeds_normal, embeds_blur = [], []
    print("Loaded model and dataset!!")
    i = 0
    torch.cuda.empty_cache()
    time.sleep(0.2)     # giving some time to clear!!
    embeds, embeds_blur = [], []
    
    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        image_tensors = prepare_input(config, sample["image"]).unsqueeze(0)

        if torch.cuda.is_available():  # half is not compatible in cpu implementation.
            image_tensors = image_tensors.half()
            image_tensors = image_tensors.to("cuda")

        output = pretrained_model.encoder(image_tensors).last_hidden_state.squeeze(0)
        image_tensors_blur = prepare_input(config, sample["image"], add_blur=True).unsqueeze(0)

        if torch.cuda.is_available():  # half is not compatible in cpu implementation.
            image_tensors_blur = image_tensors_blur.half()
            image_tensors_blur = image_tensors_blur.to("cuda")

        output_blur = pretrained_model.encoder(image_tensors_blur).last_hidden_state.squeeze(0)

        embeds.append(output.to('cpu').detach().numpy())
        embeds_blur.append(output_blur.to('cpu').detach().numpy())

        del image_tensors_blur
        del image_tensors
        del output
        del output_blur

        torch.cuda.empty_cache()
        time.sleep(0.2)     # giving some time to clear!!
        i += 1
        print(f"Done Plotting embeddings for {idx}!!")

        if i == 1:
            break

    plot_embeds(embeds, embeds_blur)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)

    predictions = viz(config)