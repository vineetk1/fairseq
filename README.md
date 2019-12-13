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
### Download dataset
Verify that the current working directory is *fairseq*.
```
cd examples/dialog
```
1. Go to https://fb-public.app.box.com/s/chnq60iivzv5uckpvj2n2vijlyepze6w 
1. Download *dialog-bAbI-tasks_1_.tgz* in directory *fairseq/examples/dialog*  
```
tar zxvf dialog-bAbI-tasks_1_.tgz
```
Verify that the dataset is in directory *fairseq/examples/dialog/dialog-bAbI-tasks*.   
### Convert dataset to fairseq dataset format
Verify that the current working directory is *fairseq/examples/dialog*.  
```
python3 create-fairseq-dialog-dataset.py data-bin/dialog
```
Verify that the converted dataset is in directory *fairseq/examples/dialog/fairseq-dialog-dataset/task6*.  
### Download pretrained word vectors
Verify that the current working directory is *fairseq/examples/dialog*.
```
mkdir pretrained-word-vectors
cd pretrained-word-vectors
```
1. Go to https://nlp.stanford.edu/projects/glove/
1. Download *glove.6B.zip* in directory *fairseq/examples/dialog/pretrained-word-vectors*
```
unzip glove.6B.zip
cd ../../..
```
Verify that the pretrained vectors are in directory *fairseq/examples/dialog/pretrained-word-vectors*.    
Verify that the current working directory is *fairseq*.
### Preprocess/binarize the data
```
TEXT=examples/dialog/fairseq-dialog-dataset/task6
python3 preprocess.py --task dialog_task --source-lang hmn --target-lang bot --joined-dictionary --trainpref $TEXT/task6-trn --validpref $TEXT/task6-dev --testpref $TEXT/task6-tst --destdir data-bin/dialog/task6
```
### Train a model
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --task dialog_task data-bin/dialog/task6 --arch dialog_lstm_model --save-dir checkpoints/dialog/task6 --max-tokens 8192 --required-batch-size-multiple 1 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer nag --lr-scheduler fixed --force-anneal 100 --lr 0.1 --clip-norm 0.1 --min-lr 2.47033e-200
```
**NOTE:** If a model has previously been trained then it is in the directory *checkpoints/dialog/task6/checkpoint_best.pt*
If training again to generate a new model then the previous obsolete model must be removed, otherwise training will resume from the last best checkpoint model. A brute-force method to remove the obsolete model is to remove the directory *fairseq/checkpoints* as follows:
```
rm -r checkpoints
```
### Evaluate the trained model
```
python3 dialog_generate.py --task dialog_task data-bin/dialog/task6 --path checkpoints/dialog/task6/checkpoint_best.pt --batch-size 128 --beam 3
```
Increasing beam-width to a number higher than 3 is preferable for better performance.  
The above command generates two files, namely, *failed_dialogs_stat.txt* and *passed_dialogs_stat.txt*. The *failed_dialogs_stat.txt* file has information about all the dialogs that failed, and *passed_dialogs_stat.txt* has information about all the dialogs that passed.  
