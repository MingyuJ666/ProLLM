import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd

# Argument parser
parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='./t5_model')
parser.add_argument('--data_path', type=str, default='instruction.csv')
parser.add_argument('--output_dir', type=str, default='./instructed_t5_model')
parser.add_argument('--num_train_epochs', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--train_batch_size', type=int, default=4)
parser.add_argument('--eval_batch_size', type=int, default=4)
parser.add_argument('--warmup_steps', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--logging_dir', type=str, default='./logs')
parser.add_argument('--logging_steps', type=int, default=150)
parser.add_argument('--eval_steps', type=int, default=300)
parser.add_argument('--save_steps', type=int, default=300)
parser.add_argument('--train_frac', type=float, default=0.95)

args = parser.parse_args()

# Load the T5-large model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=512)

# Define the dataset class
class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        input_sequence = self.data.iloc[idx]['input_text']
        output_sequence = self.data.iloc[idx]['output_text']
        input_encoding = self.tokenizer(input_sequence, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        output_encoding = self.tokenizer(output_sequence, padding='max_length', max_length=512, truncation=True, return_tensors='pt')

        input_ids = input_encoding['input_ids'].squeeze()
        attention_mask = input_encoding['attention_mask'].squeeze()
        label_ids = output_encoding['input_ids'].squeeze()

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label_ids': label_ids}

    def __len__(self):
        return len(self.data)

# Define the data collator function
def data_collator(batch):
    input_ids = torch.stack([example['input_ids'] for example in batch])
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    label_ids = torch.stack([example['label_ids'] for example in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label_ids}

def main(args):
    # Load the data
    preprocess_data = pd.read_csv(args.data_path)

    # Split the data into train and validation sets
    train_data = preprocess_data.sample(frac=args.train_frac, random_state=1)
    val_data = preprocess_data.drop(train_data.index)

    # Create the datasets
    train_dataset = MyDataset(train_data, tokenizer)
    val_dataset = MyDataset(val_data, tokenizer)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_steps=args.save_steps
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

if __name__ == "__main__":
    main(args)









