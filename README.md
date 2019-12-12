# Introduction
This implementation of a Transaction Bot deals with a dialog between a human user and a bot that results in one or more backend operations. The user initiates a dialog with a text query to a bot. The bot understands the user text, initiates execution of operations at the backend, and responds to the user in text. The dialog continues until, normally, the user terminates the dialog when its requests have been serviced by the bot. The implementation is built on the <a href="https://github.com/pytorch/fairseq" target="_blank">Facebook AI Research Sequence-to-Sequence Toolkit written in Python and PyTorch</a>.

The implementation is based on the paper: <a href="https://arxiv.org/pdf/1701.04024.pdf" target="_blank">Eric, M., & Manning, C. D. (2017). A copy-augmented sequence-to-sequence architecture gives good performance on task-oriented dialogue. arXiv preprint arXiv:1701.04024</a>. It includes an end-to-end trainable, LSTM-based Encoder-Decoder with Attention, and an Attention-based copy mechanism to track the dialog state.

Currently the implementation is in progress. Send me a note if you are interested in collaborating.
