import pandas as pd
import argparse
import time
from pathlib import Path
import torch
import h5py
from transformers import T5EncoderModel, T5Tokenizer

# Read amino acid sequence file
def read_sequence_file(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None)
    data.rename(columns={0: 'uniprot_id', 1: 'sequence'}, inplace=True)
    data['length'] = data['sequence'].apply(len)
    return data

def dataframe_to_fasta(dataframe, output_file):
    with open(output_file, 'w') as file:
        for index, row in dataframe.iterrows():
            sequence = row['sequence']
            sequence_id = str(row['uniprot_id'])
            file.write(f'>{sequence_id}\n{sequence}\n')

def get_T5_model(model_dir, device):
    if model_dir is not None:
        print("##########################")
        print("Loading cached model from: {}".format(model_dir))
        print("##########################")
    model = T5EncoderModel.from_pretrained(model_dir)
    vocab = T5Tokenizer.from_pretrained(model_dir)
    model = model.to(device)
    model = model.eval()
    return model, vocab

def read_fasta(fasta_path):
    sequences = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip()
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                sequences[uniprot_id] = ''
            else:
                sequences[uniprot_id] += ''.join(line.split()).upper().replace("-", "")
    return sequences

def get_embeddings(seq_path, emb_path, model_dir, device, per_protein, max_residues=10000, max_seq_len=5000, max_batch=100):
    seq_dict = read_fasta(seq_path)
    model, vocab = get_T5_model(model_dir, device)

    print('########################################')
    print('Example sequence: {}\n{}'.format(next(iter(seq_dict.keys())), next(iter(seq_dict.values()))))
    print('########################################')
    print('Total number of sequences: {}'.format(len(seq_dict)))

    avg_length = sum([len(seq) for _, seq in seq_dict.items()]) / len(seq_dict)
    n_long = sum([1 for _, seq in seq_dict.items() if len(seq) > max_seq_len])
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(seq_dict[kv[0]]), reverse=True)

    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))

    start = time.time()
    batch = list()
    emb_dict = dict()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
        seq_len = len(seq)
        seq = ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(token_encoding['input_ids']).to(device)
            attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

            try:
                with torch.no_grad():
                    embedding_repr = model(input_ids, attention_mask=attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={}). Try lowering batch size. ".format(pdb_id, seq_len) +
                      "If single sequence processing does not work, you need more vRAM to process your protein.")
                continue

            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                emb = embedding_repr.last_hidden_state[batch_idx, :s_len]

                if per_protein:
                    emb = emb.mean(dim=0)

                if len(emb_dict) == 0:
                    print("Embedded protein {} with length {} to emb. of shape: {}".format(
                        identifier, s_len, emb.shape))

                emb_dict[identifier] = emb.detach().cpu().numpy().squeeze()

    end = time.time()

    with h5py.File(str(emb_path), "w") as hf:
        for sequence_id, embedding in emb_dict.items():
            hf.create_dataset(sequence_id, data=embedding)

    print('\n############# STATS #############')
    print('Total number of embeddings: {}'.format(len(emb_dict)))
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(
        end - start, (end - start) / len(emb_dict), avg_length))
    return True

def main():
    parser = argparse.ArgumentParser(description='Process some protein sequences.')
    parser.add_argument('--sequence_file', type=str, default='./human_multiclass_protein_dictionary.tsv', help='Path to the amino acid sequence file')
    parser.add_argument('--fasta_file', type=str, default='./protein_sequence.fasta', help='Path to save the generated fasta file')
    parser.add_argument('--model_dir', type=str, default='./prot_t5_xl_uniref50', help='Path to the T5 model directory')
    parser.add_argument('--embedding_file', type=str, default='embedding_result.h5', help='Path to save the embeddings')
    parser.add_argument('--per_protein', type=int, default=1, help='Whether to derive per-protein (mean-pooled) embeddings')
    parser.add_argument('--max_residues', type=int, default=10000, help='Number of cumulative residues per batch')
    parser.add_argument('--max_seq_len', type=int, default=5000, help='Max length after which single-sequence processing is used to avoid OOM')
    parser.add_argument('--max_batch', type=int, default=100, help='Max number of sequences per single batch')

    args = parser.parse_args()

    data = read_sequence_file(args.sequence_file)
    dataframe_to_fasta(data, args.fasta_file)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(device))

    get_embeddings(
        model_dir=args.model_dir,
        seq_path=args.fasta_file,
        emb_path=args.embedding_file,
        device=device,
        per_protein=args.per_protein,
        max_residues=args.max_residues,
        max_seq_len=args.max_seq_len,
        max_batch=args.max_batch
    )

if __name__ == "__main__":
    main()
