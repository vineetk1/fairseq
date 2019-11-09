#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
# Vineet Kumar @ sioom: This file is a copy of generate.py, with
# changes made for the dialog implementation
"""
Translate pre-processed data with a trained model.
"""

import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.data import data_utils
from dialog_metrics import DialogMetrics
import textwrap


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    dlgs_metrics = DialogMetrics(
            src_dict, tgt_dict, args.remove_bpe, args.beam)
    has_target = True
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for dlgs_batch in t:
            dlgs_metrics.dialogs_batch(dlgs_batch)
            for dlgsBch_perTrn in dlgs_batch:
                dlgsBch_perTrn = utils.move_to_cuda(dlgsBch_perTrn) \
                                 if use_cuda else dlgsBch_perTrn
                if 'net_input' not in dlgsBch_perTrn:
                    continue

                prefix_tokens = None
                if args.prefix_size > 0:
                    prefix_tokens = \
                            dlgsBch_perTrn['target'][:, :args.prefix_size]

                gen_timer.start()
                hypos_perTrn = task.inference_step(
                        generator, models, dlgsBch_perTrn, prefix_tokens)
                num_generated_tokens = \
                    sum(len(hypo['tokens'])
                        for hypos_perTrn_perDlg in hypos_perTrn
                        for hypo in hypos_perTrn_perDlg)
                gen_timer.stop(num_generated_tokens)

                @torch.no_grad()
                def encode_hypos(best_hypos_seqs):
                    # encode best predicted sequence from each dialog in batch
                    hypos_seqs = data_utils.collate_tokens(best_hypos_seqs, 1)
                    hypos_seqs_len = hypos_seqs.new_tensor(
                                      [seq.numel() for seq in best_hypos_seqs])
                    for model in models:
                        model.encoder(**{'start_dlg': False,
                                         'src_tokens': hypos_seqs,
                                         'src_lengths': hypos_seqs_len})
                encode_hypos([hypos_perTrn_perDlg[0]['tokens']
                              for hypos_perTrn_perDlg in hypos_perTrn])

                dlgs_metrics.hypos_per_turn(hypos_perTrn)

                # remaining code is for Bleu Score
                for i, sample_id in enumerate(dlgs_batch[0]['id'].tolist()):
                    has_target = dlgsBch_perTrn['target'] is not None

                    # Remove padding
                    src_tokens = utils.strip_pad(dlgsBch_perTrn['net_input']['src_tokens'][i, :], tgt_dict.pad())
                    target_tokens = None
                    if has_target:
                        target_tokens = utils.strip_pad(dlgsBch_perTrn['target'][i, :], tgt_dict.pad()).int().cpu()

                    # Either retrieve the original sentences or regenerate them from tokens.
                    if align_dict is not None:
                        src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                        target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                    else:
                        if src_dict is not None:
                            src_str = src_dict.string(src_tokens, args.remove_bpe)
                        else:
                            src_str = ""
                        if has_target:
                            target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                    # Process top predictions
                    for j, hypo in enumerate(hypos_perTrn[i][:min(len(hypos_perTrn), args.nbest)]):
                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                            hypo_tokens=hypo['tokens'].int().cpu(),
                            src_str=src_str,
                            alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                            align_dict=align_dict,
                            tgt_dict=tgt_dict,
                            remove_bpe=args.remove_bpe,
                        )

                        # Score only the top hypothesis
                        if has_target and j == 0:
                            if align_dict is not None or args.remove_bpe is not None:
                                # Convert back to tokens for evaluation with unk replacement and/or without BPE
                                target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                            if hasattr(scorer, 'add_string'):
                                scorer.add_string(target_str, hypo_str)
                            else:
                                scorer.add(target_tokens, hypo_tokens)

                wps_meter.update(num_generated_tokens)
                t.log({'wps': round(wps_meter.avg)})

    strng = '\nStatistics on the test\n----------------------'
    dlgs_metrics.write_out(strng, write_to=["stdout", "file"])

    strng = 'Translated {} dialogs with {} turns and {} tokens using beam={} in {:.1f}s ({:.2f} dialogs/s, {:.2f} turns/s, {:.2f} tokens/s)'.format(
             dlgs_metrics.num_dlgs(), dlgs_metrics.num_trns(), gen_timer.n,
             args.beam, gen_timer.sum, dlgs_metrics.num_dlgs()/gen_timer.sum,
             dlgs_metrics.num_trns()/gen_timer.sum, 1./gen_timer.avg)
    dlgs_metrics.write_out(strng, write_to=["stdout", "file"], bullet=True,
                           next_lines_manual_indent=False)

    dlgs_metrics.print_stats()

    if has_target:
        strng = 'Generate {} with beam={}: {}'.format(
                            args.gen_subset, args.beam, scorer.result_string())
        dlgs_metrics.write_out(strng, write_to=["stdout", "file"], bullet=True,
                               next_lines_manual_indent=False)

    return scorer


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
