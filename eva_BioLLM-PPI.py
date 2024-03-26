import pandas as pd
import numpy as np
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import argparse

parser = argparse.ArgumentParser(description='Run T5 model for text generation.')
parser.add_argument('--model_path', type=str, required=True, help='Path to the T5 model directory.')
parser.add_argument('--csv_path', type=str, default='Human_test_10.csv', help='Path to the test CSV file.')

args = parser.parse_args()

model_path = args.model_path
csv_path = args.csv_path

preds1 = []
expec1 = []
input1 = []

correct = 0
total = 0
count = 0

model_path = '/root/autodl-tmp/KGLLM_REMAKE/Human-7_embedding/checkpoint-100'
model = T5ForConditionalGeneration.from_pretrained(model_path)

tokenizer = T5Tokenizer.from_pretrained(model_path)

test_df = pd.read_csv(csv_path)
test_df = test_df.sample(100)

for index, row in test_df.iterrows():
    input_text = row['input_text']
    expected_output = row['output_text']

    # print("The expected is {}".format(expected_output))
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    # 进行预测并解码输出
    generation_config = {
        'max_new_tokens': 1500
    }
    outputs = model.generate(input_ids, **generation_config)

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('The output is {}'.format(output_text))

    # if len(output_text) > 0:
    #     output_text = output_text.split()[0]

    output_text_new = output_text.split()[-1].replace(".", "").replace("_", "").replace(",","")
    expected_output_new = expected_output.split()[-1].replace(".", "").replace("_", "").replace(",","")
    print("The expected is {}".format(expected_output_new))
    print('The output is {}'.format(output_text_new))
    print("---------------------------------------------------------------------")


    preds1.append(output_text_new)
    expec1.append(expected_output_new)
    input1.append(input_text.split())
    if expected_output_new == output_text_new:
        correct += 1
    total += 1

accuracy = correct / total
print(f"Accuracy: {accuracy}")
print(preds1)
print(expec1)
print("-------------------------------------------------------------")



# 计算F1分数
f1 = f1_score(expec1, preds1, average='micro')  # Use 'micro', 'macro', or 'weighted' based on your preference.
print(f"F1 Score: {f1}")

print(f"F1 Score: {f1}")

print("-------------------------------------------------------------")