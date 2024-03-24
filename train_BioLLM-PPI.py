import argparse
import json
import pandas as pd
from collections import deque
import csv
import random
from sklearn.model_selection import train_test_split
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd


parser = argparse.ArgumentParser(description='Fine-tune a model on a dataset')
parser.add_argument('--model_dir', type=str, default='/root/autodl-tmp/KGLLM_REMAKE/newT5', help='Directory of the model')
parser.add_argument('--tokenizer_dir', type=str, default='/root/autodl-tmp/KGLLM_REMAKE/newT5', help='Directory of the tokenizer')
parser.add_argument('--data_file', type=str, default='SHS27K_train_10.csv', help='Dataset CSV file')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and evaluation')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--output_dir', type=str, default='./SHS27K-7_embedding', help='Output directory for saving models and logs')
args = parser.parse_args()



# Load the FLAN-T5 model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, model_max_length=512)




# Define the dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        # Get the input and output sequences
        input_sequence = self.data.iloc[idx]['input_text']
        output_sequence = self.data.iloc[idx]['output_text']
        # print('output_sequence is {}'.format(output_sequence))
        input_encoding = tokenizer(input_sequence, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        output_encoding = tokenizer(output_sequence, padding='max_length', max_length=512, truncation=True, return_tensors='pt')

        # Get the input IDs, attention mask, and label IDs from the encodings
        # -------------------------------------------------------------------
        input_ids = input_encoding['input_ids'].squeeze()
        attention_mask = input_encoding['attention_mask'].squeeze()
        label_ids = output_encoding['input_ids'].squeeze()
        # print('label_ids: {}'.format(label_ids))

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label_ids': label_ids}
        # -------------------------------------------------------------------

    def __len__(self):
        return len(self.data)

# Define the data collator function
def data_collator(batch):
    input_ids = torch.stack([example['input_ids'] for example in batch])
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    label_ids = torch.stack([example['label_ids'] for example in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label_ids}

# Load the data
preprocess_data = pd.read_csv(args.data_file)

# Split the data into train and validation sets
train_data = preprocess_data.sample(frac=1, random_state=1)
val_data = preprocess_data.drop(train_data.index)

# Create the datasets
train_dataset = MyDataset(train_data)
val_dataset = MyDataset(val_data)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_epochs,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy='steps',
    eval_steps=50000,
    save_steps=100,
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()