Example usage
```
# NOTE: All paths are relative to the fairseq directory

# (1) Download dataset:
# Verify that the current working directory is fairseq.
$ cd examples/dialog/
# Download dataset manually:
# (a) Go to https://fb-public.app.box.com/s/chnq60iivzv5uckpvj2n2vijlyepze6w 
# (b) Download dialog-bAbI-tasks_1_.tgz in directory fairseq/examples/dialog
$ tar zxvf dialog-bAbI-tasks_1_.tgz
$ rm dialog-bAbI-tasks_1_.tgz
# Verify that the dataset is in directory fairseq/examples/dialog/dialog-bAbI-tasks.

# (2) Convert dataset to fairseq dataset format
# Verify that the current working directory is fairseq/examples/dialog.
$ python3 create-fairseq-dialog-dataset.py data-bin/dialog
# Verify that the converted dataset is in directory fairseq/examples/dialog/fairseq-dialog-dataset/task6.

# (3) Download pretrained word vectors
# Verify that the current working directory is fairseq/examples/dialog.
$ mkdir pretrained-word-vectors
$ cd pretrained-word-vectors
# Download manually:
# (a) Go to https://nlp.stanford.edu/projects/glove/
# (b) Download glove.6B.zip in directory fairseq/examples/dialog/pretrained-word-vectors
$ unzip glove.6B.zip
$ rm glove.6B.zip
$ cd ../../..
# Verify that the pretrained vectors are in directory fairseq/examples/dialog/pretrained-word-vectors.
# Verify that the current working directory is fairseq.

# (4) Preprocess/binarize the data
$ TEXT=examples/dialog/fairseq-dialog-dataset/task6
$ python3 preprocess.py --task dialog_task --source-lang hmn --target-lang bot --joined-dictionary --trainpref $TEXT/task6-trn --validpref $TEXT/task6-dev --testpref $TEXT/task6-tst --destdir data-bin/dialog/task6

# (5) Train the model
$ CUDA_VISIBLE_DEVICES=0 python3 -m pdb train.py --task dialog_task data-bin/dialog/task6 --arch dialog_lstm_model --save-dir checkpoints/dialog/task6 --max-tokens 8192 --required-batch-size-multiple 1 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer nag --lr-scheduler fixed --force-anneal 100 --lr 0.1 --clip-norm 0.1 --min-lr 2.47033e-200
# NOTE: If a model has previously been trained then it is in the directory checkpoints/dialog/task6/checkpoint_best.pt If training again to generate a new model then the previous obsolete model must be removed, otherwise training will resume from the last best checkpoint model. A brute-force method to remove the obsolete model is to remove the directory fairseq/checkpoints as follows:
$ rm -r checkpoints

# (6) Evaluate the trained model
$ python3 -m pdb dialog_generate.py --task dialog_task data-bin/dialog/task6 --path checkpoints/dialog/task6/checkpoint_best.pt --batch-size 128 --beam 3 
