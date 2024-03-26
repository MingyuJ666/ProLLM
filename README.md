
# BioLLM-PPI: Protein Chain-of-Thoughts for LLM-based Protein-Protein Interaction Prediction
This repo presents implementation of the **BioLLM-PPI🧬** 
<div align=center><img src="pic/pic1.png" width="100%" height="100%" /></div>



We present **Biological Large Language Model for Protein-Protein Interaction prediction**, abbreviated as **BioLLM-PPI**. This innovative framework leverages the advanced capabilities of Large Language Models (LLMs) to interpret and analyze protein sequences and interactions through a natural language processing lens. 

## Key Features of BioLLM-PPI

- **Protein Chain of Thought (ProCoT) Method**: BioLLM-PPI introduces the Protein Chain of Thought (ProCoT) method, transforming the complex, structured data of protein interactions into intuitive, natural language prompts. 

- **Enhanced Predictive Accuracy**: This approach not only facilitates a deeper understanding of protein functions and interactions but also enhances the model's predictive accuracy by incorporating protein-specific embeddings and instruction fine-tuning on domain-specific datasets.
<div align=center><img src="pic/Flow.png" width="100%" height="100%" /></div>

## Usage

  

0. Clone this repo

  

```

git clone https://github.com/jmyissb/BioLLM.git

```

1. Download [SHS27K, SHS148K, STRING] from (https://drive.google.com/file/d/1hJVrQXddB9JK68z7jlIcLfd9AmTWwgJr/view?usp=drive_link)
   Download [Human] from (https://drive.google.com/drive/folders/1hT_lAZUB0p-8AuV7x8BCa8cltlhSQpmQ?usp=drive_link)

  

2. Download Flan-T5-large pretrained checkpoint via this ...
