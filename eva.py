import argparse
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser(description="T5 Model Inference Script")
parser.add_argument("--model_path", type=str, default="your_model_path",required=True, help="Path to the pre-trained model")
parser.add_argument("--test_df", type=str, default="test_data_path",required=True, help="Path to the test CSV file")


args = parser.parse_args()

def main(model_path, test_df_path):

    preds1 = []
    expec1 = []
    input1 = []
    correct = 0
    total = 0


    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)


    test_df = pd.read_csv(test_df_path)


    generation_config = {
        'max_new_tokens': 1500,
        'num_beams': 5,
        'early_stopping': True
    }


    for index, row in test_df.iterrows():
        input_text = row['input_text']
        expected_output = row['output_text']


        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        try:

            outputs = model.generate(input_ids, **generation_config)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during generation at index {index}: {e}")
            continue


        output_text = output_text.split()[-1].replace(".", "").replace("_", "")
        expected_output_new = expected_output.split()[-1].replace(".", "").replace("_", "")


        print(f"Expected: {expected_output_new}")
        print(f"Output: {output_text}")

        preds1.append(output_text)
        expec1.append(expected_output_new)
        input1.append(input_text.split())


        if expected_output_new == output_text:
            correct += 1
        total += 1


    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f}")


    # f1 = f1_score(expec1, preds1, average='weighted')
    # print(f"F1 Score: {f1:.4f}")


main(args.model_path, args.test_df)