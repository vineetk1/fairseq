'''
Vineet Kumar, sioom corp
'''

import torch
from fairseq import utils
from collections import Counter
from itertools import takewhile


class DialogMetrics(object):
    def __init__(self, src_dict, tgt_dict, remove_bpe, beam_size):
        # a turn (trn) of a dialog (dlg) has src-seq, tgt-seq, hypo-seqs
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.remove_bpe = remove_bpe
        self.beam_size = beam_size
        self.count = Counter()
        self.max_num_trns = 0

    def dialogs_batch(self, dlgs_smpl):
        # dlgs_smpl: list of (# of turns in dialogs) dicts
        assert len(dlgs_smpl) == dlgs_smpl[0]['maxNumSentencesInDialog']
        assert dlgs_smpl[0]['net_input']['src_tokens'].size(0) ==\
            dlgs_smpl[0]['numDialogs']
        assert dlgs_smpl[0]['target'].size(0) == dlgs_smpl[0]['numDialogs']
        assert not hasattr(self, 'dlgs_batch')
        assert not hasattr(self, 'hypos_batch')
        self.dlgs_batch = dlgs_smpl
        self.hypos_batch = []

    def _update_metrics_for_dlg(self, dlg_trn_rslt):
        # dlg_trn_rslt: ((# of turns in dlg) x (# of hypos = beam size)) of
        #    turn_result; where (1) a turn does not include empty padded seqs
        #    at end of dlg, and (2) turn_result = True if tgt-seq matches
        #    highest probability hypo seq

        self.count[f'num_dlgs'] += 1
        num_trns_in_dlg = dlg_trn_rslt.size(0)
        self.count[f'num_trns'] += num_trns_in_dlg
        if num_trns_in_dlg > self.max_num_trns:
            self.max_num_trns = num_trns_in_dlg
        # dialog passes if all turn-results of highest probability hypo passes
        if dlg_trn_rslt[:, 0].all():
            self.count[f'num_trns_pass'] += num_trns_in_dlg
            self.count[f'num_dlgs_pass'] += 1
            self.count[f'num_dlgs_pass {num_trns_in_dlg}'] += 1
        else:
            self.count[f'num_dlgs_fail'] += 1
            self.count[f'num_trns_pass'] += sum(
                                    [1 for trns in dlg_trn_rslt[:, 0] if trns])
            # find which hypo has max # of consecutive turns that pass,
            #    counting turns from beginning
            hypos_num_consec_trns_pass = [
                    sum([1 for trn_rslt in takewhile(lambda x: x, hypo_rslt)])
                    for hypo_rslt in dlg_trn_rslt.transpose(0, 1)]
            num_consec_trns_pass, hypo_num = \
                torch.tensor(hypos_num_consec_trns_pass).topk(1)
            self.count[f'{hypo_num.item()}'] += 1
            self.count[f'{hypo_num.item()} {num_consec_trns_pass.item()} {num_trns_in_dlg}'] += 1

    def hypos_per_turn(self, hypos_per_trn):
        # hypos_per_trn: (list of (# dialogs) lists) x (list of (beam size) dicts)
        assert len(hypos_per_trn) == self.dlgs_batch[0]['numDialogs']
        assert len(hypos_per_trn[0]) == self.beam_size
        # hypos_batch: list of (# of turns in dialogs) hypos_per_trn
        self.hypos_batch.append(hypos_per_trn)
        if len(self.hypos_batch) ==\
                self.dlgs_batch[0]['maxNumSentencesInDialog']:
            for dlg_num in torch.arange(self.dlgs_batch[0]['numDialogs']):
                trn_rslts = []
                for trn_num in torch.arange(
                        self.dlgs_batch[0]['maxNumSentencesInDialog']):
                    src_tokens = utils.strip_pad(self.dlgs_batch[trn_num]
                                 ['net_input']['src_tokens'][dlg_num],
                                 self.src_dict.pad())
                    assert src_tokens.size(0)    # seq has at least an eos
                    if src_tokens.size(0) == 1:
                        assert src_tokens.item() == self.src_dict.eos()
                        break               # remaining seqs are also empty
                    tgt_tokens = utils.strip_pad(self.dlgs_batch[trn_num]
                                 ['target'][dlg_num], self.tgt_dict.pad())
                    assert tgt_tokens.size(0)    # seq has at least an eos
                    trn_rslts.append([torch.equal(tgt_tokens,
                              hypo['tokens'].cpu()) for hypo in
                              self.hypos_batch[trn_num][dlg_num]])
                assert len(trn_rslts)   # dialog must have at least 1 turn
                self._update_metrics_for_dlg(torch.tensor(trn_rslts))
            del self.dlgs_batch
            del self.hypos_batch

    def print_stats(self):
        for num_trns_in_dlg in range(1, self.max_num_trns+1):
            self.count[f'num_dlgs_pass'] += 1
            self.count[f'num_dlgs_pass {num_trns_in_dlg}'] += 1
            self.count[f'num_dlgs_fail'] += 1
            self.count[f'num_trns_pass'] += 1
            for hypo_num in range(self.beam_size):
                self.count[f'{hypo_num}'] += 1
                for num_consec_trns_pass in range(1, self.max_num_trns+1):
                    self.count[f'{hypo_num} {num_consec_trns_pass} {num_trns_in_dlg}'] += 1

        print('\n Statistics on the test')
        print('--------------------------')

        # Statistics on dialogs that passed
        num_dlgs_pass = self.count['num_dlgs_pass']
        print('** % number of dialogs that passed = ({}/{} x 100) = {}%'.format(
            num_dlgs_pass, self.count['num_dlgs'],
            num_dlgs_pass/self.count['num_dlgs'] * 100))
        if num_dlgs_pass:
            first_time = True
            for num_trns_in_dlg in range(1, self.max_num_trns+1):
                cnt_num_trns_in_dlg = self.count[f'num_dlgs_pass {num_trns_in_dlg}']
                if cnt_num_trns_in_dlg:
                    if first_time:
                        print(f'    ** ((# of turns in dialog: # of occurrences) = ({num_trns_in_dlg}: {cnt_num_trns_in_dlg})', end="")
                        first_time = False
                    else:
                        print(f', ({num_trns_in_dlg}: {cnt_num_trns_in_dlg})', end="")
            print()

        # Statistics on turns of dialogs
        print('** % number of turns that passed = ({}/{} x 100) = {}%'.format(
            self.count['num_trns_pass'], self.count['num_trns'],
            self.count['num_trns_pass']/self.count['num_trns'] * 100))

        # Statistics on dialogs that failed
        num_dlgs_fail = self.count['num_dlgs_fail']
        print('** % number of dialogs that failed = ({}/{} x 100) = {}%'.format(
            num_dlgs_fail, self.count['num_dlgs'],
            num_dlgs_fail/self.count['num_dlgs'] * 100))
        if num_dlgs_fail:
            print(f'    hypo 0 sequence has highest probability whereas hypo {self.beam_size-1} sequence has lowest probability')
            for hypo_num in range(self.beam_size):
                cnt_hypo_num = self.count[f'{hypo_num}']
                if cnt_hypo_num:
                    print(f'    ** hypo = {hypo_num} , # of occurrences = {cnt_hypo_num}')
                    first_time = True
                    for num_trns_in_dlg in range(1, self.max_num_trns+1):
                        for num_consec_trns_pass in range(1, num_trns_in_dlg+1):
                            cnt_consecTrns_per_trnsInDlg = self.count[f'{hypo_num} {num_consec_trns_pass} {num_trns_in_dlg}']
                            if cnt_consecTrns_per_trnsInDlg:
                                if first_time:
                                    print(f'        ** (# of consecutive turns that passed, counting from beginning of dialog / # of turns in dialog: # of occurrences) = ({num_consec_trns_pass}/{num_trns_in_dlg}: {cnt_consecTrns_per_trnsInDlg})', end="")
                                    first_time = False
                                else:
                                    print(f', ({num_consec_trns_pass}/{num_trns_in_dlg}: {cnt_consecTrns_per_trnsInDlg})', end="")
                    print()

