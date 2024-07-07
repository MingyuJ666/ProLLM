import argparse
import h5py
import pickle
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Make sure the ProTrans embeddings are prepared')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the H5 file containing ProTrans embeddings.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the pre-trained tokenizer.')
    parser.add_argument('--save_directory', type=str, required=True, help='Directory to save the updated tokenizer and model.')
    parser.add_argument('--freeze_embeddings', action='store_true', help='Whether to freeze the new embeddings or not.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    embedding_dict = {}

    with h5py.File(args.file_path, 'r') as hf:
        for key in hf.keys():
            dataset = hf[key]
            data = dataset[:]
            embedding_dict[key] = data

    with open('./embedding_dict.pkl', 'wb') as f:
        pickle.dump(embedding_dict, f)

    with open('./embedding_dict.pkl', 'rb') as f:
        result_dict = pickle.load(f)

    # Extract and process protein list
    protein_list = list(result_dict.keys())
    protein_list_new = [x.split('_')[-1] for x in protein_list]

    # Print processed vocabulary list
    print("Processed Vocabulary List:")
    for token in protein_list_new:
        print(token)

    # Load pre-trained model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, model_max_length=512)

    # New tokens and embeddings
    new_tokens = protein_list_new
    desired_shape = (len(result_dict), 1024)
    new_tokens_embeddings = torch.empty(desired_shape, dtype=torch.float32)

    # Fill tensor with embeddings
    for i, value in enumerate(result_dict.values()):
        new_tokens_embeddings[i] = torch.tensor(value, dtype=torch.float32)



    print("New Tokens and their Embeddings:")
    for token, embedding in zip(new_tokens, new_tokens_embeddings):
        print(f"Token: {token}")
        print(f"Embedding: {embedding}")

    tokenizer.add_tokens(new_tokens)

    model.resize_token_embeddings(len(tokenizer))

    embeddings = model.get_input_embeddings().weight.data
    embeddings[-len(new_tokens):] = new_tokens_embeddings

    if args.freeze_embeddings:
        for param in model.get_input_embeddings().parameters():
            param.requires_grad = False
            param.data[-len(new_tokens):].requires_grad = False


    # Save updated tokenizer and model
    tokenizer.save_pretrained(args.save_directory)
    model.save_pretrained(args.save_directory)

if __name__ == '__main__':
    main()
