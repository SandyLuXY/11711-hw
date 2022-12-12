# 11711-hw2

## Group Member: 
### Tinglong Zhu (tinglonz), Haozhe Zhang (haozhez2), Xinyu Lu (xinyulu2)

## Teamwork:
Haozhe, Tinglong and Xinyu contributed equally to the assignment. For the collecting raw data part, Tinglong wrote the crawler to download PDFs from the ACL website, and Haozhe wrote the script to extract sentences line-by-line and tokenize the data. For the data annotation part, Haozhe annotated 5 articles, Tinglong annotated 1 article and Xinyu annotated 6 articles. For the model training part, Tinglong developed the bert-base-NER+LC, bert-base-NER+TDNN, SciBERT+LC and SciBERT+TDNN models, and he also tried using CNN layers. Xinyu developed the bert-base-NER+LC and bert-base-uncased+LC models. Tinglong organized the code and wrote the instructions of how to run the code. For the report, Haozhe wrote the data collection and comparative analysis parts, Tinglong wrote the model details part, and Xinyu wrote the data collection and experiments part.

## Environment Setup
```bash
pip install spacy PyPDF2
pip install git+https://github.com/huggingface/transformers
pip install -U scikit-learn
```
In our project, we conduct experiments on PyTorch 1.12.1 with cuda11.3

## Data preparation
### crawling data
```bash
./crawler.sh
```
one can change the url corresponding to their needs in the script.

### PDF to CoNLL
```bash
python python pdf2coll.py pdf_pth conll_sav_pth sentence_sav_pth
```
Here the three arguments are all directory path. The script will 
automatically scan through the dir to find pdf files.

### Training data preparation
```bash
python conll2npy_train.py sci
```
One can change the dataset/datafile's name in the scripts.
Here sci refers to SciBERT model, if the user wanted to use bert-base-NER
to extract embedding, one can replace `sci` with `ner`

### Evaluation data preparation
```bash
python conll2npy_eval.py sci True
```
Here the `True` is for whether the testing data is for private test set evaluation 
on explain board.

## Training
```bash
python main.py [encoder] [clf]
```
* Here the encoder denotes the encoder used to extract embedding, it could be 
`sci` or `ner`
* clf denotes the classifier structure, it could be `Linear` or `TDNN`

The training outputs, including the loss for each epoch and predicted training/evaluation labels 
generated after each epoch will be saved in the output
directory, while save the checkpoints for model weights in the checkpoints dir.
One can easily load the saved weights for the classifier from the checkpoints.
## Generate evaluation for private test set
```bash
python prediction2conll.py
```
## Generate evaluation for evaluation set
```bash
python prediction2conll_eval.py
```
One can change the saved prediction and output filename path in these two python scripts

## Analysis & Graphs
One can refer to the `confusion_matrix.ipynb`.