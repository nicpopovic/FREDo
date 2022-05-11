# Few-Shot Document-Level Relation Extraction

Code and data for the NAACL 2022 paper [Few-Shot Document-Level Relation Extraction](https://arxiv.org/abs/2205.02048)

# Table of Contents
1. [Accessing the data (FREDo benchmark tasks)](#accessing-the-data)
2. [Dependencies](#dependencies)
3. [Reproducing Results (+ link to trained models)](#reproducing-results)
4. [Relevant Code per Section in Paper](#relevant-code-per-section-in-paper)
4. [Bibtex for Citing](#cite)

# Accessing the data

## Download Sampled Episodes
If you are only interested in the raw data for the tasks, you can directly download sampled episodes here: https://drive.google.com/drive/folders/1PuJSJxqZP4ijxFSBZZ6Fmc0SgR2S8pYU?usp=sharing (test_episodes.zip \[~120 MB DL, ~550MB on disk\] + train_episodes.zip \[~330MB DL, ~1.4GB on disk\])

In our paper and code we evaluate *macro* F1 scores across relation types.

## Training and Development Data
- data/\[train, dev, test_docred, test_scierc\].json contain all annotated documents used for training and testing.
- data/*_indices.json contain sampled episodes (only the indices of the documents and which relations are to be annotated/extracted).

You can either use our pipeline or export all episodes as json files to use in your own pipeline.

Test episode sampling is called in train.py (Lines 72+73).

## Export Episodes to JSON
If you are evaluating your own model, please use the test episodes as sampled by us.
You can run export_test_episodes.py to export all episodes for use with your own pipeline.


# Dependencies

Dependencies:
- torch (1.8.0)
- transformers (4.18.0)
- tqdm (4.64.0)
- wandb (0.12.16)

```
python3 -m venv venv
source venv/bin/activate
pip install wheel
pip install -U setuptools
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers tqdm wandb
```

# Reproducing Results

In order to reproduce the results from the paper, you can download the models here: https://drive.google.com/drive/folders/1eulLgrGiOwSZawoOGytA6fJZ-b3Rcrob?usp=sharing

The following are the commands for reporducing the results in the paper. Click on "Expected output" for detailed results.

## BASELINE 1-DOC
```
python train.py --support_docs_eval 1 --model dlmnav+sie+sbn --num_epochs 0 --use_markers False
```

<details>
  <summary>Expected output</summary>

  ```
3nX3120z6X
---- INDOMAIN TEST EVAL -----
type         precision      recall          f1    support   
P361              0.20        1.79        0.37    893       
P279              0.02        5.15        0.03    136       
P102              0.42       14.49        0.82    428       
P17               3.76       11.08        5.61    59312     
P35               0.29       12.61        0.56    222       
P495              0.41       10.08        0.79    1081      
P463              0.09        1.62        0.18    493       
P3373             0.18       16.43        0.36    645       
P118              0.13        8.30        0.25    253       
P39               0.03        2.63        0.07    38        
P140              0.12       13.97        0.24    458       
P272              0.07       10.00        0.13    160       
P674              0.28        7.76        0.54    348       
P25               0.16       19.28        0.31    83        
P364              0.03       16.92        0.06    65        
P1001             0.14       10.13        0.27    375       
P194              0.07        8.56        0.14    222       
P582              0.07        3.96        0.14    101       
-                    -           -           -    -         
macro             0.36        9.71        0.60
---- SCIERC TEST EVAL -----
type         precision      recall          f1    support   
USED-FOR          3.60        3.82        3.71    42142     
PART-OF           0.46        1.90        0.74    2529      
CONJUNCTION        1.92        5.24        2.81    6414      
EVALUATE-FOR        1.22        2.79        1.70    4223      
HYPONYM-OF        1.30        3.64        1.92    4482      
COMPARE           0.54        2.49        0.89    2328      
FEATURE-OF        0.33        1.30        0.53    2465      
-                    -           -           -    -         
macro             1.34        3.03        1.76
```


</details>


## BASELINE 3-DOC
```
python train.py --support_docs_eval 3 --model dlmnav+sie+sbn --num_epochs 0 --use_markers False
```

<details>
  <summary>Expected output</summary>

```
G7BwlKpec6
---- INDOMAIN TEST EVAL -----
type         precision      recall          f1    support   
P17               5.19        6.11        5.61    33229     
P361              0.50        1.80        0.78    887       
P272              0.14       20.00        0.27    55        
P495              0.69        6.35        1.24    850       
P674              0.45        8.59        0.86    128       
P35               0.32        2.70        0.56    111       
P102              0.76       15.77        1.45    279       
P140              0.23        2.34        0.42    171       
P364              0.32       19.23        0.63    78        
P463              0.34        3.11        0.61    193       
P3373             0.37        7.88        0.71    241       
P194              0.21        8.62        0.41    116       
P1001             0.20        7.78        0.39    167       
P279              0.04        8.47        0.08    59        
P25               0.03        5.26        0.05    19        
P118              0.38       13.83        0.75    94        
P582              0.14        5.56        0.27    36        
P39               0.42       50.00        0.84    2         
-                    -           -           -    -         
macro             0.60       10.75        0.89
---- SCIERC TEST EVAL -----
type         precision      recall          f1    support   
PART-OF           0.61        1.65        0.89    1335      
USED-FOR          3.98        3.02        3.44    14548     
HYPONYM-OF        2.16        1.56        1.81    2311      
CONJUNCTION        3.07        4.53        3.66    3332      
COMPARE           1.09        3.36        1.64    1162      
EVALUATE-FOR        1.29        1.89        1.53    2382      
FEATURE-OF        0.68        1.21        0.88    1235      
-                    -           -           -    -         
macro             1.84        2.46        1.98
```

</details>

## DLMNAV 1-DOC

```
python train.py --support_docs_eval 1 --model dlmnav --num_epochs 0 --load_checkpoint checkpoints/best_dlmnav.pt
```

<details>
  <summary>Expected output</summary>

```
38BE1Nj8Pu
loading model from checkpoints/best_dlmnav.pt
---- INDOMAIN TEST EVAL -----
type         precision      recall          f1    support   
P279              0.37        2.94        0.66    136       
P17              30.40        2.38        4.41    59312     
P463              3.82        8.11        5.19    493       
P495             11.43       16.47       13.50    1081      
P674              4.45       23.56        7.49    348       
P361              3.33        5.04        4.01    893       
P364              1.76       36.92        3.36    65        
P39               0.27       15.79        0.52    38        
P35               5.06       27.48        8.54    222       
P3373             7.12       21.24       10.67    645       
P118             15.54       62.06       24.86    253       
P1001             1.93        6.40        2.96    375       
P194              3.90       30.63        6.92    222       
P25               2.03       45.78        3.89    83        
P102             10.61       14.72       12.33    428       
P272              4.47       47.50        8.17    160       
P582              3.13       27.72        5.62    101       
P140              9.24       10.04        9.62    458       
-                    -           -           -    -         
macro             6.60       22.49        7.37
---- SCIERC TEST EVAL -----
type         precision      recall          f1    support   
USED-FOR          2.54        0.04        0.08    42142     
PART-OF           0.59        0.36        0.44    2529      
FEATURE-OF        0.32        0.16        0.22    2465      
CONJUNCTION        1.06        0.25        0.40    6414      
EVALUATE-FOR        0.91        0.19        0.31    4223      
HYPONYM-OF        6.25        2.57        3.64    4482      
COMPARE           1.31        0.69        0.90    2328      
-                    -           -           -    -         
macro             1.85        0.61        0.86
```

</details>

## DLMNAV 3-DOC

```
python train.py --support_docs_eval 3 --model dlmnav --num_epochs 0 --load_checkpoint checkpoints/best_dlmnav.pt
```

<details>
  <summary>Expected output</summary>

```
eASm3VD2Bv
loading model from checkpoints/best_dlmnav.pt
---- INDOMAIN TEST EVAL -----
type         precision      recall          f1    support   
P17              32.74        1.66        3.17    33229     
P361              4.51        4.51        4.51    887       
P674              5.35       31.25        9.14    128       
P495             11.91       14.94       13.26    850       
P140              7.08        8.77        7.83    171       
P39               0.00        0.00        0.00    2         
P463              3.45        6.22        4.44    193       
P194              6.65       31.03       10.96    116       
P35               4.64       27.03        7.93    111       
P364             10.21       62.82       17.56    78        
P272              4.55       58.18        8.44    55        
P582              4.36       38.89        7.84    36        
P1001             1.67        2.40        1.97    167       
P118             16.35       45.74       24.09    94        
P279              1.05        5.08        1.73    59        
P25               1.78       47.37        3.43    19        
P102             12.29       15.41       13.67    279       
P3373            10.77       26.14       15.25    241       
-                    -           -           -    -         
macro             7.74       23.75        8.62
---- SCIERC TEST EVAL -----
type         precision      recall          f1    support   
USED-FOR          0.00        0.00        0.00    14548     
PART-OF           0.00        0.00        0.00    1335      
HYPONYM-OF        8.76        1.69        2.83    2311      
CONJUNCTION        0.65        0.06        0.11    3332      
COMPARE           0.94        0.26        0.41    1162      
EVALUATE-FOR        1.73        0.25        0.44    2382      
FEATURE-OF        0.00        0.00        0.00    1235      
-                    -           -           -    -         
macro             1.73        0.32        0.54
```

</details>

## DLMNAV+SIE 1-DOC

```
python train.py --support_docs_eval 1 --model dlmnav+sie --num_epochs 0 --load_checkpoint checkpoints/best_sie_sbn.pt
```

<details>
  <summary>Expected output</summary>

```
vxMctq1z4F
loading model from checkpoints/best_sie_sbn.pt
---- INDOMAIN TEST EVAL -----
type         precision      recall          f1    support   
P279              0.52        2.94        0.89    136       
P17              22.84        3.44        5.98    59312     
P463              3.27       11.56        5.10    493       
P495              5.14        2.68        3.53    1081      
P674              5.90       22.99        9.38    348       
P1001             6.67       17.60        9.68    375       
P35               6.17       22.52        9.68    222       
P39               0.33       18.42        0.64    38        
P361              3.04        2.80        2.92    893       
P364              2.60       35.38        4.85    65        
P25               1.85       48.19        3.56    83        
P3373             5.87       25.89        9.58    645       
P118             11.25       77.08       19.63    253       
P102              6.71       26.17       10.68    428       
P582              2.59       17.82        4.52    101       
P194              4.78       13.96        7.13    222       
P272              5.31       44.38        9.49    160       
P140              6.44       13.76        8.77    458       
-                    -           -           -    -         
macro             5.63       22.64        7.00
---- SCIERC TEST EVAL -----
type         precision      recall          f1    support   
USED-FOR          6.35        1.73        2.72    42142     
PART-OF           0.68        0.51        0.59    2529      
CONJUNCTION        1.61        0.72        0.99    6414      
FEATURE-OF        0.16        0.12        0.14    2465      
EVALUATE-FOR        1.86        1.18        1.45    4223      
HYPONYM-OF        3.76        1.43        2.07    4482      
COMPARE           2.36        1.42        1.77    2328      
-                    -           -           -    -         
macro             2.40        1.02        1.39
```

</details>

## DLMNAV+SIE 3-DOC

```
python train.py --support_docs_eval 3 --model dlmnav+sie --num_epochs 0 --load_checkpoint checkpoints/best_sie_sbn.pt
```

<details>
  <summary>Expected output</summary>

```
Ta4ae83La8
loading model from checkpoints/best_sie_sbn.pt
---- INDOMAIN TEST EVAL -----
type         precision      recall          f1    support   
P361              3.92        3.49        3.69    887       
P674              5.29       42.97        9.42    128       
P17              24.60        5.30        8.72    33229     
P495              3.57        2.47        2.92    850       
P140              4.95       13.45        7.23    171       
P102              6.39       25.09       10.18    279       
P364              4.61       60.26        8.56    78        
P463              3.39       14.51        5.49    193       
P39               0.05       50.00        0.11    2         
P194              6.32       24.14       10.02    116       
P1001             5.08       23.35        8.35    167       
P279              1.24       22.03        2.35    59        
P118              7.23       78.72       13.24    94        
P272              3.62       63.64        6.85    55        
P35               3.95       14.41        6.20    111       
P3373             6.77       32.78       11.22    241       
P582              2.58       36.11        4.82    36        
P25               1.21       73.68        2.37    19        
-                    -           -           -    -         
macro             5.26       32.58        6.76
---- SCIERC TEST EVAL -----
type         precision      recall          f1    support   
USED-FOR          6.37        3.62        4.61    14548     
HYPONYM-OF        5.14        2.34        3.21    2311      
FEATURE-OF        0.36        0.40        0.38    1235      
PART-OF           0.74        0.82        0.78    1335      
EVALUATE-FOR        1.57        1.22        1.37    2382      
CONJUNCTION        3.14        1.98        2.43    3332      
COMPARE           2.78        2.58        2.68    1162      
-                    -           -           -    -         
macro             2.87        1.85        2.21

```

</details>

## DLMNAV+SIE+SBN 1-DOC

```
python train.py --support_docs_eval 1 --model dlmnav+sie+sbn --num_epochs 0 --load_checkpoint checkpoints/best_sie_sbn.pt
```

<details>
  <summary>Expected output</summary>

```
HXGmfwnstC
loading model from checkpoints/best_sie_sbn.pt
---- INDOMAIN TEST EVAL -----
type         precision      recall          f1    support   
P361              0.33        3.14        0.59    893       
P279              0.04        1.47        0.07    136       
P17               8.89       20.26       12.35    59312     
P35               0.72       23.42        1.40    222       
P495              1.47       25.16        2.78    1081      
P463              0.36       14.00        0.71    493       
P674              0.40       31.03        0.79    348       
P3373             0.98       43.72        1.92    645       
P1001             0.62       29.60        1.22    375       
P194              0.45       32.88        0.88    222       
P118              0.42       34.78        0.84    253       
P272              0.69       36.25        1.36    160       
P39               0.01       13.16        0.02    38        
P582              0.23       15.84        0.46    101       
P102              0.90       49.30        1.76    428       
P140              0.23       12.66        0.44    458       
P364              0.06       15.38        0.12    65        
P25               0.40       21.69        0.78    83        
-                    -           -           -    -         
macro             0.96       23.54        1.58
---- SCIERC TEST EVAL -----
type         precision      recall          f1    support   
USED-FOR          4.62        5.14        4.86    42142     
PART-OF           0.58        1.82        0.88    2529      
CONJUNCTION        3.30        8.53        4.76    6414      
FEATURE-OF        0.39        1.74        0.63    2465      
EVALUATE-FOR        2.08        4.10        2.76    4223      
HYPONYM-OF        2.22        4.46        2.97    4482      
COMPARE           1.74        4.34        2.48    2328      
-                    -           -           -    -         
macro             2.13        4.30        2.76
```

</details>

## DLMNAV+SIE+SBN 3-DOC

```
python train.py --support_docs_eval 3 --model dlmnav+sie+sbn --num_epochs 0 --load_checkpoint checkpoints/best_sie_sbn.pt
```

<details>
  <summary>Expected output</summary>

```
AmcBdjBC9r
loading model from checkpoints/best_sie_sbn.pt
---- INDOMAIN TEST EVAL -----
type         precision      recall          f1    support   
P17              11.96       17.46       14.20    33229     
P361              0.66        3.83        1.13    887       
P495              2.63       16.12        4.52    850       
P272              0.88       32.73        1.72    55        
P279              0.11        5.08        0.22    59        
P140              1.24       12.28        2.25    171       
P102              2.10       48.03        4.02    279       
P39               0.01       50.00        0.01    2         
P674              1.24       25.00        2.37    128       
P364              0.66       34.62        1.30    78        
P463              0.33        6.22        0.63    193       
P35               0.41        5.41        0.77    111       
P194              1.43       30.17        2.73    116       
P1001             1.88       29.94        3.54    167       
P118              0.62       28.72        1.21    94        
P3373             3.96       42.32        7.25    241       
P582              0.34       13.89        0.66    36        
P25               0.37       21.05        0.72    19        
-                    -           -           -    -         
macro             1.71       23.49        2.74
---- SCIERC TEST EVAL -----
type         precision      recall          f1    support   
USED-FOR          6.06        5.69        5.87    14548     
HYPONYM-OF        4.48        3.81        4.12    2311      
FEATURE-OF        0.46        0.97        0.62    1235      
PART-OF           0.85        1.87        1.17    1335      
CONJUNCTION        5.63        9.12        6.96    3332      
EVALUATE-FOR        3.12        3.57        3.33    2382      
COMPARE           2.83        4.65        3.52    1162      
-                    -           -           -    -         
macro             3.35        4.24        3.66
```

</details>

# Relevant Code per Section in Paper

## Section 4.3 Test Episode Sampling

Covered by parse_test() in src/data.py (lines 138-324)

## Section 5.1.1 Baseline

Covered in src/models/base_model.py
Covered in src/models/dlmnav_sbn.py

Run model using:
```
python train.py --query_docs_train 3 --query_docs_eval 3 --support_docs_train 1 --support_docs_eval 1 --model dlmnav+sie+sbn --num_epochs 0 --use_markers False
```
## Section 5.1.2 DLMNAV

Covered in src/models/base_model.py
Covered in src/models/dlmnav_sbn.py

Train model using:
```
python train.py --query_docs_train 3 --query_docs_eval 3 --support_docs_train 1 --support_docs_eval 1 --model dlmnav
```
## Section 5.1.3 SIE

Covered in src/models/base_model.py
Covered in src/models/dlmnav_sie.py

Train model using:
```
python train.py --query_docs_train 3 --query_docs_eval 3 --support_docs_train 1 --support_docs_eval 1 --model dlmnav+sie
```
## Section 5.1.4 SBN

Covered in src/models/base_model.py
Covered in src/models/dlmnav_sbn.py

Train model using:
```
python train.py --query_docs_train 3 --query_docs_eval 3 --support_docs_train 1 --support_docs_eval 1 --model dlmnav+sie+sbn
```
## Section 5.2 Sampling Training & Development Episodes

Covered by parse_episodes() in src/data.py (lines 144-324)

Parameters for test sets are found in train.py (lines 70-71)


# Cite
If you use the benchmark or the code, please cite this paper:

```
@inproceedings{popovic_fsdlre_2022, 
 title = "{F}ew-{S}hot {D}ocument-{L}evel {R}elation {E}xtraction", 
 author = "Popovic, Nicholas and Färber, Michael",
 booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies", 
 year = "2022"
}
```