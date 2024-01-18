# Data-Driven Knowledge-Aware Inference of Private Information in Continuous Double Auctions  

This repository is the specific implementation of our accepted paper on AAAI-2024.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Knowledge-Aware Inference

To train our KATE model on CDA data, run the command:
```train
python knowledgeAware-KATE.py --file_name 'Filtered_data_CDA_trans_IR' --split_idx 1 --sigma 0.82623 --save_path <save root path> 
```


## Evaluation

To evaluate learning-based inference models on CDA data, run the command:

```eval
python eval_learningMethods_on_testdata.py --file_name 'Filtered_data_CDA_trans_IR' --split_idx 1 --model_path <trained model root>
```

## Others

To split the data into different train/valid/test, run the command:
```run
python split_data_into_trainValid.py 
```

