# Step 1: Generate the ProTrans embedding
```
python generate_embedding.py --sequence_file ./human_multiclass_protein_dictionary.tsv --fasta_file ./protein_sequence.fasta --model_dir ./prot_t5_xl_uniref50 --embedding_file embedding_result.h5

```

The "sequence_file" is the protein sequence file, the "fasta_file" is another format for saving the protein sequence file, the "model_dir" is the directory of the ProTrans model, and the "embedding_file" is the file for saving the embeddings.


# Step 2: Do the embedding replacement
```
python embedding_replacement.py --file_path ./embedding_result.h5 --model_path ./flan-t5-large --tokenizer_path ./flan-t5-large --save_directory ./new_embedding_t5

```


"file_path" is the embedding to be replaced, "model_path" is the model to be replaced, "tokenizer_path" is the tokenizer to be replaced, and "save_directory" is the save location for the new model.
