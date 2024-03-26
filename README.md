
# BioLLM-PPI: Protein Chain-of-Thoughts for LLM-based Protein-Protein Interaction Prediction
This repo presents implementation of the **BioLLM-PPI🧬** 
<div align=center><img src="pic/pic1.png" width="100%" height="100%" /></div>



We present **Biological Large Language Model for Protein-Protein Interaction prediction**, abbreviated as **BioLLM-PPI**. This innovative framework leverages the advanced capabilities of Large Language Models (LLMs) to interpret and analyze protein sequences and interactions through a natural language processing lens. 

## Key Features of BioLLM-PPI🔑

- **Protein Chain of Thought (ProCoT) Method**: BioLLM-PPI introduces the Protein Chain of Thought (ProCoT) method, transforming the complex, structured data of protein interactions into intuitive, natural language prompts. 

- **Enhanced Predictive Accuracy**: This approach not only facilitates a deeper understanding of protein functions and interactions but also enhances the model's predictive accuracy by incorporating protein-specific embeddings and instruction fine-tuning on domain-specific datasets.
<div align=center><img src="pic/Flow.png" width="100%" height="100%" /></div>

## Requirements

See the requirements.txt

## Usage🧰

  

0. Clone this repo

  

```

git clone https://github.com/jmyissb/BioLLM.git

```
### train🧑‍💻
1. Download [SHS27K, SHS148K, STRING](https://drive.google.com/file/d/1hJVrQXddB9JK68z7jlIcLfd9AmTWwgJr/view?usp=drive_link) and [Human](https://drive.google.com/drive/folders/1hT_lAZUB0p-8AuV7x8BCa8cltlhSQpmQ?usp=drive_link)

2. Preprocess the dataset into Protein Chain of Thought (ProCoT) by the [dataset]_preprocess.py
```

python [dataset]_preprocess.py

```

3. Do training through train_BioLLM-PPI.py , make sure the location of model and tokenizer in right place. Feel free to change 'num_epochs', 'batch_size' and 'learning_rate'.
```

python train_BioLLM-PPI.py 

```

6. Evaluate through eva_BioLLM-PPI.py , the model to evaluate should be the location of the training model from step 3.

```

python eva_BioLLM-PPI.py 

```

## Checkpoint 
See: [Google Drive link](https://drive.google.com/drive/folders/1mHGdE0x57pX2ie6U4BJxZg5w3KL2SWa7?usp=drive_link).
