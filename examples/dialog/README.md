Example usage:
```
# NOTE: All paths are relative to the fairseq directory

# (1) Download dataset as follows:
$ cd examples/dialog/
# Download dataset manually as follows:
# (a) Go to https://fb-public.app.box.com/s/chnq60iivzv5uckpvj2n2vijlyepze6w 
# (b) Download dialog-bAbI-tasks_1_.tgz in directory fairseq/examples/dialog
$ tar zxvf dialog-bAbI-tasks_1_.tgz

# (2) Modify the dataset so it is understood by fairseq framework 
$ python3 create-fairseq-dialog-dataset.py data-bin/dialog

# (3) Download pretrained word vectors
$ mkdir pretrained-word-vectors
$ cd pretrained-word-vectors
$ Go to https://nlp.stanford.edu/projects/glove/ and download glove.6B.zip; it will take some time to download
$ unzip glove.6B.zip
$ cd ../..

# (4) Binarize the dataset:
$ TEXT=examples/dialog/fairseq-dialog-dataset/task6
$ python3 preprocess.py --task dialog_task --source-lang hmn --target-lang bot --joined-dictionary --trainpref $TEXT/task6-trn --validpref $TEXT/task6-dev --testpref $TEXT/task6-tst --destdir data-bin/dialog/task6

# (5) Train the model (better for a single GPU setup):
# ***NOTE*** if Training must be started from the beginning then the checkpoint files
# in the directory "checkpoints/dialog/task1" must be removed, otherwise training
# will resume from the last best checkpoint model
$ rm -r checkpoints
$ CUDA_VISIBLE_DEVICES=0 python3 -m pdb train.py --task dialog_task data-bin/dialog/task6 --arch dialog_lstm_model --save-dir checkpoints/dialog/task6 --max-tokens 8192 --required-batch-size-multiple 1 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer nag --lr-scheduler fixed --force-anneal 200 --lr 0.1 --clip-norm 0.1 --min-lr 2.47033e-200

# (6) Generate:
# Increase beam-width if system does not run out of CUDA memory
$ python3 -m pdb dialog_generate.py --task dialog_task data-bin/dialog/task6 --path checkpoints/dialog/task6/checkpoint_best.pt --batch-size 128 --beam 3 
