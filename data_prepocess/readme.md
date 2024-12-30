# Change the PPI data into ProCoT format QA dataset to finetune the LLMs

## 1.Put the corresponding dataset into the folder of the same name. Ex: Human dataset to ./Human


### 
How to do the multi-label?
We have a parser 'multi_label', it can control the prepocess process of our ProCoT data for further fine-tuning.

```
python your_script.py --input_file Human_PPI.tsv --train_file Human_train.csv --test_file Human_test.csv --total_train 30000 --total_test 5000 --graph_size 10 --random_state 42 --multi_label
```
