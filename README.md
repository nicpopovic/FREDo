# Few-Shot Document-Level Relation Extraction

Code and data for our paper [Few-Shot Document-Level Relation Extraction](https://arxiv.org/abs/2205.02048)

## Accessing the data
### Training and Development Data
data/\[train, dev, test_docred, test_scierc\].json contain all annotated documents used for training and testing.

data/*_indices.json contain sampled episodes (only the indices of the documents and which relations are to be annotated/extracted).

You can either use our pipeline or export all episodes as json files to use in your own pipeline.

### Export Episodes to JSON
If you are evaluating your own model, please use the test episodes sampled by us.
You can run export_test_episodes.py to export all episodes for use with your own pipeline.

## Environment Setup
```
python3 -m venv venv
source venv/bin/activate
pip install wheel
pip install -U setuptools
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers tqdm wandb
```

## Reproduce Results

In order to reproduce the results from the paper, you can download the models here: https://drive.google.com/drive/folders/1eulLgrGiOwSZawoOGytA6fJZ-b3Rcrob?usp=sharing

```
# BASELINE 3-DOC
python train.py --support_docs_eval 3 --model dlmnav+sie+sbn --num_epochs 0 --use_markers False
# BASELINE 1-DOC
python train.py --support_docs_eval 1 --model dlmnav+sie+sbn --num_epochs 0 --use_markers False


# DLMNAV 3-DOC
python train.py --support_docs_eval 3 --model dlmnav --num_epochs 0 --load_checkpoint checkpoints/best_dlmnav.pt
# DLMNAV 1-DOC
python train.py --support_docs_eval 1 --model dlmnav --num_epochs 0 --load_checkpoint checkpoints/best_dlmnav.pt


# DLMNAV+SIE 3-DOC
python train.py --support_docs_eval 3 --model dlmnav+sie --num_epochs 0 --load_checkpoint checkpoints/best_sie_sbn.pt
# DLMNAV+SIE 1-DOC
python train.py --support_docs_eval 1 --model dlmnav+sie --num_epochs 0 --load_checkpoint checkpoints/best_sie_sbn.pt


# DLMNAV+SIE+SBN 3-DOC
python train.py --support_docs_eval 3 --model dlmnav+sie+sbn --num_epochs 0 --load_checkpoint checkpoints/best_sie_sbn.pt
# DLMNAV+SIE+SBN 1-DOC
python train.py --support_docs_eval 1 --model dlmnav+sie+sbn --num_epochs 0 --load_checkpoint checkpoints/best_sie_sbn.pt
```

## Relevant Code per Section in Paper

### Section 4.3 Test Episode Sampling

Covered by parse_test() in src/data.py (lines 138-324)

Parameters for test sets are found in train.py (lines 72-73)

### Section 5.1.1 Baseline

Covered in src/models/base_model.py
Covered in src/models/dlmnav_sbn.py

Run using:
```
python train.py --query_docs_train 3 --query_docs_eval 3 --support_docs_train 1 --support_docs_eval 1 --model dlmnav+sie+sbn --num_epochs 0 --use_markers False
```
### Section 5.1.2 DLMNAV

Covered in src/models/base_model.py
Covered in src/models/dlmnav_sbn.py

Run using:
```
python train.py --query_docs_train 3 --query_docs_eval 3 --support_docs_train 1 --support_docs_eval 1 --model dlmnav
```
### Section 5.1.3 SIE

Covered in src/models/base_model.py
Covered in src/models/dlmnav_sie.py

Run using:
```
python train.py --query_docs_train 3 --query_docs_eval 3 --support_docs_train 1 --support_docs_eval 1 --model dlmnav+sie
```
### Section 5.1.4 SBN

Covered in src/models/base_model.py
Covered in src/models/dlmnav_sbn.py

Run using:
```
python train.py --query_docs_train 3 --query_docs_eval 3 --support_docs_train 1 --support_docs_eval 1 --model dlmnav+sie+sbn
```
### Section 5.2 Sampling Training & Development Episodes

Covered by parse_episodes() in src/data.py (lines 144-324)

Parameters for test sets are found in train.py (lines 70-71)


## Cite
If you use the benchmark or the code, please cite this paper:

```
@inproceedings{popovic_fsdlre_2022, 
 title = "{F}ew-{S}hot {D}ocument-{L}evel {R}elation {E}xtraction", 
 author = "Popovic, Nicholas and Färber, Michael",
 booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies", 
 year = "2022"
}
```