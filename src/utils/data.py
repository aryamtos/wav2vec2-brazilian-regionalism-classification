from sklearn.model_selection import train_test_split
import os
import sys
import torchaudio
import IPython.display as ipd
import numpy as np
import torch
import pandas as pd
from datasets import Dataset


def load_data(directory, max_samples_per_class=None):
    data = []
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)

        if os.path.isdir(class_path):
            class_data = []
            
            for file_name in os.listdir(class_path):
                if file_name.endswith(".wav"):
                    file_path = os.path.join(class_path, file_name)
                    name_file = os.path.splitext(file_name)[0]

                    class_data.append({
                        "path": file_path,
                        "label": class_name,
                        "name_file": name_file
                    })
            if max_samples_per_class and len(class_data) > max_samples_per_class:
                class_data = class_data[:max_samples_per_class]

            data.extend(class_data)

    return data

def preprocess_data(directory:str,samples:int):
    train_directory = directory
    max_samples_per_class = samples
    data_train = load_data(train_directory, max_samples_per_class)
    df = pd.DataFrame(data_train)
    print(f"Step 0: {len(df)}")
    
    df = df.dropna(subset=["path"])
    
    print(f"Step 1: {len(df)}")
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    return df 

def split_and_concat_data(df, test_size=0.1, random_state=101):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["label"])
    
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    concatenated_df = pd.concat([train_df, test_df])
    
    return train_df, test_df, concatenated_df




