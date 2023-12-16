"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
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


def viz(args):
    checkpoint = "naver-clova-ix/donut-base"

    processor = DonutProcessor.from_pretrained(checkpoint)
    pretrained_model = VisionEncoderDecoderModel.from_pretrained(checkpoint)

    # pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda:4")

    pretrained_model.eval()
    dataset = load_dataset(args.dataset_name_or_paths[0], split="validation")
    embeds_normal, embeds_blur = [], []
    print("Loaded model and dataset!!")
    i = 0

    tsne = TSNE(2, verbose=1)
    cmap = cm.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(8,8))
    embeds, embeds_blur = [], []
    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        image_tensors = prepare_input(config, sample["image"]).unsqueeze(0)

        if torch.cuda.is_available():  # half is not compatible in cpu implementation.
            image_tensors = image_tensors.half()
            image_tensors = image_tensors.to("cuda:4")

        output = pretrained_model.encoder(image_tensors).last_hidden_state.squeeze(0)
        print(output.shape)
        # embeds_normal.append(output)

        image_tensors_blur = prepare_input(config, sample["image"], add_blur=True).unsqueeze(0)

        if torch.cuda.is_available():  # half is not compatible in cpu implementation.
            image_tensors_blur = image_tensors_blur.half()
            image_tensors_blur = image_tensors_blur.to("cuda:4")

        output_blur = pretrained_model.encoder(image_tensors_blur).last_hidden_state.squeeze(0)
        print(output_blur.shape)
        # embeds_blur.append(output_blur)
        # print("Generated :: ", idx)
        # import pdb;pdb.set_trace()
        
        print("Plotting embeddings!!")
        # embeds_normal, embeds_blur = np.array(embeds_normal), np.array(embeds_blur)
        # import pdb;pdb.set_trace();
        embeds.append(output.to('cpu').detach().numpy())
        embeds_blur.append(output_blur.to('cpu').detach().numpy())

        del image_tensors_blur
        del image_tensors
        del output
        del output_blur

        torch.cuda.empty_cache()
        time.sleep(0.2)     # giving some time to clear!!
        i += 1
        if i == 10:
            break

    for i in range(len(embeds)):
        fig, ax = plt.subplots(figsize=(8,8))
        proj_normal = tsne.fit_transform(np.array(embeds[i]))
        proj_blur = tsne.fit_transform(np.array(embeds_blur[i]))
        # proj_normal = tsne.fit_transform(embeds_normal)
        # proj_blur = tsne.fit_transform(embeds_blur)

        ax.scatter(proj_normal[:,0],proj_normal[:,1], c=np.array(cmap(0)).reshape(1,4), label = "Normal" ,alpha=0.5)
        ax.scatter(proj_blur[:,0],proj_blur[:,1], c=np.array(cmap(1)).reshape(1,4), label = "With Blur" ,alpha=0.5)

        ax.legend(fontsize='large', markerscale=2)
        plt.savefig(f'modie_embeddings_initial_{i}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)

    predictions = viz(config)
