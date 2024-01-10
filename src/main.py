
import numpy as np


from pathlib import Path
from tqdm import tqdm
import argparse
import torchaudio
from sklearn.model_selection import train_test_split
import os
import sys
#import librosa
from utils.data import preprocess_data, split_and_concat_data
from model.speech_classifier import Wav2Vec2ForSpeechClassification
from model.training import training
from model.metrics import compute_metrics
from model.data_collactor import DataCollatorCTCWithPadding
from dataloader import get_label_list,create_config, create_feature_extractor,speech_file_to_array_fn,prepare_dataset
import IPython.display as ipd
import numpy as np
import torch
import pandas as pd
from datasets import Dataset
from transformers import Trainer



def train(args):

    dataset_train = args.dataset_train
    dataset_val = args.dataset_val
    max_samples = args.max_sample_train
    model_name = args.model_name
    batch_size = args.batch_size
    data_train=preprocess_data(dataset_train,max_samples)
    data_val= preprocess_data(dataset_val,2254)
    train_df, test_df, train_df_concat = split_and_concat_data(data_train)
    train_dv, test_dv, dev_df = split_and_concat_data(preprocess_data(data_val))

    train_dataset = Dataset.from_pandas(train_df_concat)
    eval_dataset = Dataset.from_pandas(dev_df)
    print(train_dataset)
    print(eval_dataset)
    pooling_mode="mean"
    input_column = "path"
    output_column = "label"

    label_list, num_labels =  get_label_list(train_dataset,output_column)
    config = create_config(model_name, label_list, pooling_mode)
    feature_extractor = create_feature_extractor(model_name)
    train_dataset = train_dataset.map(
        lambda batch: speech_file_to_array_fn(batch, seconds_stop=10, s_rate=16_000),
        remove_columns=train_dataset.column_names,
        num_proc=4)
    
    eval_dataset = eval_dataset.map(
        lambda batch: speech_file_to_array_fn(batch, seconds_stop=10, s_rate=16_000),
        remove_columns=eval_dataset.column_names,
        num_proc=4,
        )
    
    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset(batch, feature_extractor),
        remove_columns=train_dataset.column_names,
        batch_size=batch_size,
        batched=True,
        num_proc=4,
    )

    eval_dataset = eval_dataset.map(
        lambda batch: prepare_dataset(batch, feature_extractor),
        remove_columns=eval_dataset.column_names,
        batch_size=batch_size,
        batched=True,
        num_proc=4,
    )

    data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, padding=True)
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_name,
        config=config,
        )
    
    training_args=training()
    model.freeze_feature_extractor()
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=feature_extractor,)
    trainer.train()


if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='Example of parser')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_train',type=str,required=True,help='Directory of train set.')
    parser_train.add_argument('--dataset_val',type=str,required=True,help='Directory of validation.')
    parser_train.add_argument('--max_samples',type=int,required=True,help='Directory of max samples.')
    parser_train.add_argument('--model_name',type=str,required=True,help='Model path checkpoint.')

