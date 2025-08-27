
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from transformers import DistilBertTokenizerFast

# Dataset loading
col_names = ['ID', 'label', 'statement', 'subject', 'speaker', 'speaker\'s title', 'state', 'party', 'barely true', 'false', 'half true', 'mostly true', 'pants on fire', 'context', 'justification']
train_df = pd.read_csv('data/train2.tsv', sep='\t', names=col_names)
test_df = pd.read_csv('data/test2.tsv', sep='\t', names=col_names)
val_df = pd.read_csv('data/val2.tsv', sep='\t', names=col_names)

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)
val_df = val_df.fillna(0)

train_df.head()

train_df['label'].value_counts()

print(f'Number of datapoints are {len(train_df)}')
print(f'Shape of df is {train_df.shape}')
train_df.info()

train_df["label"].value_counts().head(7).plot(kind = 'pie', autopct='%1.1f%%', figsize=(4, 4)).legend(bbox_to_anchor=(1, 1))

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', do_lower_case=True, use_auth_token=None)

# Encoding labels
def encode_labels(df):
    df['label'] = df['label'].map({
        'true': 1, 'mostly-true': 1, 'half-true': 1,
        'false': 0, 'barely-true': 0, 'pants-fire': 0
    })
    return df

train_df = encode_labels(train_df)
val_df = encode_labels(val_df)
test_df = encode_labels(test_df)

label_counts = train_df['label'].value_counts()

print("Label distribution in the Train set:")
print(f"False: {label_counts.get(0, 0)} example")
print(f"True: {label_counts.get(1, 0)} example")

# Combine text e metadata
def combine_text_and_metadata(df):
    meta = []
    for i in range(len(df)):
        subject = df['subject'][i] if df['subject'][i] != 0 else 'None'
        speaker = df['speaker'][i] if df['speaker'][i] != 0 else 'None'
        job = df["speaker's title"][i] if df["speaker's title"][i] != 0 else 'None'
        state = df['state'][i] if df['state'][i] != 0 else 'None'
        party = df['party'][i] if df['party'][i] != 0 else 'None'
        context = df['context'][i] if df['context'][i] != 0 else 'None'
        meta_text = f"{subject} {speaker} {job} {state} {party} {context}"
        meta.append(meta_text)
    df['combined_text'] = df['statement'].astype(str) + " " + pd.Series(meta)
    return df

train_df = combine_text_and_metadata(train_df)
val_df = combine_text_and_metadata(val_df)
test_df = combine_text_and_metadata(test_df)