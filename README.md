# Keyphrase Extraction using BERT (Semeval 2017, Task 10)

Deep Keyphrase extraction using BERT.

## Usage

**It takes a lot of computational power to work correctly. I kept the file in a way that only can work in colab environment**    

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lMAZXUi3IzrFcG5f-HZCu0AACWVXGC0A?usp=sharing)    
[![Open In Drive](https://img.shields.io/badge/googledrive-Open%20in%20Drive-yellowgreen)](https://drive.google.com/drive/folders/1SfK8YJ1mPyBt0Hz61vOqzMnAmvdnheg-?usp=sharing)    


**If you want to run this in your local machine follow this steps:**    

1. Clone this repository and install `transformers` with this command `pip3 install transformers`
2. From `bert` repo, untar the weights (rename their weight dump file to `pytorch_model.bin`) and vocab file into a new folder `model`. (can be skipped for limitation, leads to poor performance)
3. Change the parameters accordingly in `experiments/base_model/params.json`. We recommend keeping batch size of 4 and sequence length of 512, with 6 epochs, if GPU's VRAM is around 11 GB.
4. For training, run the command `python train.py --data_dir data/task1/ --bert_model_dir model/ --model_dir experiments/base_model`
5. For eval, run the command, `python evaluate.py --data_dir data/task1/ --bert_model_dir model/ --model_dir experiments/base_model --restore_file best`

## Results

### Subtask 1: Keyphrase Boundary Identification

We used IO format here. Unlike original BERT repo, we only use a simple linear layer on top of token embeddings.

On test set, we got:

1. **F1 score**: 0.34
2. **Precision**: 0.45
3. **Recall**: 0.27
4. **Support**: 921

<!--
### Subtask 2: Keyphrase Classification

We used BIO format here. Overall F1 score was 0.4981 on test set.

|          | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Process  | 0.4734    | 0.5207 | 0.4959   | 870     |
| Material | 0.4958    | 0.6617 | 0.5669   | 807     |
| Task     | 0.2125    | 0.2537 | 0.2313   | 201     |
| Avg      | 0.4551    | 0.5527 | 0.4981   | 1878    |
-->


### Future Work

1. Some tokens have more than one annotations. We did not consider multi-label classification.
2. We only considered a linear layer on top of BERT embeddings. We need to see whether SciBERT + BiLSTM + CRF makes a difference.



## Credits

1. SciBERT: https://github.com/allenai/scibert
2. HuggingFace: https://github.com/huggingface/pytorch-pretrained-BERT
3. PyTorch NER: https://github.com/lemonhu/NER-BERT-pytorch
4. BERT: https://github.com/google-research/bert
