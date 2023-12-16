# -*- coding: utf-8 -*-

from datasets import load_dataset
from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel
from sconf import Config
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from nltk import edit_distance
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, BaseFinetuning
from torch.utils.data import DataLoader
from utils.dataset import DonutDataset
from pathlib import Path
import re
import numpy as np
import math
import torch



dataset = load_dataset("naver-clova-ix/cord-v2")
config = Config("config/train_cord.yaml")

# update image_size of the encoder
# during pre-training, a larger image size was used
encdecconfig = VisionEncoderDecoderConfig.from_pretrained(config.pretrained_model_name_or_path)
encdecconfig.encoder.image_size = config.input_size # (height, width)

# update max_length of the decoder (for generation)
encdecconfig.decoder.max_length = config.max_length
# TODO we should actually update max_position_embeddings and interpolate the pre-trained ones:
# https://github.com/clovaai/donut/blob/0acc65a85d140852b8d9928565f0f6b2d98dc088/donut/model.py#L602

# instantiate the model with our custom config, as well as the processor. 
# Make sure that all pre-trained weights are correctly loaded (a warning would tell you if that's not the case)
processor = DonutProcessor.from_pretrained(config.pretrained_model_name_or_path)
model = VisionEncoderDecoderModel.from_pretrained(config.pretrained_model_name_or_path, config=encdecconfig)

# we update some settings which differ from pretraining; namely the size of the images + no rotation required
# source: https://github.com/clovaai/donut/blob/master/config/train_cord.yaml
processor.image_processor.size = config.input_size[::-1] # should be (width, height)
processor.image_processor.do_align_long_axis = False

train_dataset = DonutDataset("naver-clova-ix/cord-v2", config=config,
                             split="train", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
                             sort_json_key=config.sort_json_key, # cord dataset is preprocessed, so no need for this
                             processor = processor,
                             model = model,
                             )

val_dataset = DonutDataset("naver-clova-ix/cord-v2", config=config,
                             split="validation", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
                             sort_json_key=config.sort_json_key, # cord dataset is preprocessed, so no need for this
                             processor = processor,
                             model = model,
                             )


# the vocab size attribute stays constants (might be a bit unintuitive - but doesn't include special tokens)
print("Original number of tokens:", processor.tokenizer.vocab_size)
print("Number of tokens after adding special tokens:", len(processor.tokenizer))

pixel_values, labels, target_sequence = train_dataset[0]
print("Pixel values shape:: ",pixel_values.shape)

# let's print the labels (the first 30 token ID's)
for id in labels.tolist()[:30]:
  if id != -100:
    print(processor.decode([id]))
  else:
    print(id)

# let's check the corresponding target sequence, as a string
print("Target Sequence: ", target_sequence)

model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s_cord-v2>'])[0]

# sanity check
print("Pad token ID:", processor.decode([model.config.pad_token_id]))
print("Decoder start token ID:", processor.decode([model.config.decoder_start_token_id]))

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

    def training_step(self, batch, batch_idx):
        pixel_values, labels, _ = batch

        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, labels, answers = batch
        batch_size = pixel_values.shape[0]
        # we feed the prompt to the model
        decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id, device=self.device)

        outputs = self.model.generate(pixel_values,
                                   decoder_input_ids=decoder_input_ids,
                                   max_length=self.config.max_length,
                                   early_stopping=True,
                                   pad_token_id=self.processor.tokenizer.pad_token_id,
                                   eos_token_id=self.processor.tokenizer.eos_token_id,
                                   use_cache=True,
                                   num_beams=1,
                                   bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                   return_dict_in_generate=True,)

        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            predictions.append(seq)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            # NOT NEEDED ANYMORE
            # answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))

        return scores

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

class DecoderFreeze(BaseFinetuning):
    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.model.decoder)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch in (0,1):
            for name, param in pl_module.model.decoder.named_parameters():
                print(f"Layer {name} has requires_grad={param.requires_grad}.")

model_module = DonutModelPLModule(config, processor, model)
early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")
callbacks = []
if not config.freeze_decoder:
    checkpoint_callback = ModelCheckpoint(dirpath=config.result_path, filename='modie_e2e_{epoch}-{val_loss:.2f}', save_top_k=1, save_last=False, mode='min')
else:
    checkpoint_callback = ModelCheckpoint(dirpath=config.result_path, filename='modie_e2e_frozen_decoder_{epoch}-{val_loss:.2f}', save_top_k=1, save_last=False, mode='min')
    decoderfreeze = DecoderFreeze()
    callbacks.append(decoderfreeze)

callbacks = [early_stop_callback, checkpoint_callback] + callbacks

trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.max_epochs,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        gradient_clip_val=config.gradient_clip_val,
        precision=16, # we'll use mixed precision
        num_sanity_val_steps=0,
        callbacks=callbacks,
)

trainer.fit(model_module)