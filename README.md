# Introduction
This implementation of a Transaction Bot deals with a dialog between a human user and a bot that results in one or more backend operations. The user initiates a dialog with a text query to a bot. The bot understands the user text, initiates execution of operations at the backend, and responds to the user in text. The dialog continues until, normally, the user terminates the dialog when its requests have been serviced by the bot. The implementation is built on the <a href="https://github.com/pytorch/fairseq" target="_blank">Facebook AI Research Sequence-to-Sequence Toolkit written in Python and PyTorch</a>.

The implementation is based on the paper: <a href="https://arxiv.org/pdf/1701.04024.pdf" target="_blank">Eric, M., & Manning, C. D. (2017). A copy-augmented sequence-to-sequence architecture gives good performance on task-oriented dialogue. arXiv preprint arXiv:1701.04024</a>. It includes an end-to-end trainable, LSTM-based Encoder-Decoder with Attention.
# Requirements
* PyTorch version >= 1.2.0
* Python version >= 3.6
# Installation
```
git clone https://github.com/vineetk1/fairseq.git
cd fairseq
pip install --editable .
```
# Training a new model
###   Download dataset
Check that the current working directory is fairseq/examples/dialog
1. Go to https://fb-public.app.box.com/s/chnq60iivzv5uckpvj2n2vijlyepze6w 
1. Download dialog-bAbI-tasks_1_.tgz in directory fairseq/examples/dialog
```
tar zxvf dialog-bAbI-tasks_1_.tgz
```
Check that the dataset is in directory fairseq/examples/dialog/dialog-bAbI-tasks
###   Convert dataset to fairseq format
Check that the current working directory is fairseq/examples/dialog
```
python3 create-fairseq-dialog-dataset.py data-bin/dialog
```
Check that the converted dataset is in directory fairseq/examples/dialog/fairseq-dialog-dataset/task6
###   Download pretrained word vectors
python3 create-fairseq-dialog-dataset.py data-bin/dialog
```
mkdir pretrained-word-vectors
cd pretrained-word-vectors
```
1. Go to https://nlp.stanford.edu/projects/glove/
1. Download glove.6B.zip; it will take some time to download
```
unzip glove.6B.zip
cd ../../..
```
Check that the pretrained vectors are in directory fairseq/examples/dialog/pretrained-word-vectors
Check that the current working directory is fairseq
