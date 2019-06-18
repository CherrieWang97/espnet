#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import logging
import random
import time

import numpy as np
from sklearn.model_selection import train_test_split
from .vocab import phoneme_to_sequence

_batches_per_group = 64

class DataFeeder(object):
    def __init__(self, src_path, trg_path, args, speaker_file=None):
        super(DataFeeder, self).__init__()
        self.args = args
        self._test_offset = 0

        self._speaker_table = {}
        if speaker_file is not None:
            for line in open(speaker_file):
                filename, speaker_id = line.strip().split('\t')
                self._speaker_table[filename] = int(speaker_id)

        self._src_mel_dir = os.path.join(src_path, 'mels')
        self._trg_mel_dir = os.path.join(trg_path, 'mels')
        self._trg_linear_dir = os.path.join(trg_path, 'linear')

        # load src_data
        with open(os.path.join(src_path, args.input), encoding='utf-8') as f:
            self._metadata = [line.strip().split('|') for line in f]
            frame_shift_ms = args.hop_size / args.sample_rate
            src_hours = sum([int(x[2]) for x in self._metadata]) * frame_shift_ms / (3600)
            trg_hours = sum([int(x[4]) for x in self._metadata]) * frame_shift_ms / (3600)
            logging.info('Loaded metadata for {} examples source audio: {:.2f} hours, target audio: {:.2f} hours'.format(
                len(self._metadata), src_hours, trg_hours))

        # Train test split
        if args.test_size is None:
            assert args.test_batches is not None

        test_size = (args.test_size if args.test_size is not None
                     else args.test_batches * args.batch_size)
        indices = np.arange(len(self._metadata))
        train_indices, test_indices = train_test_split(indices,
                                                       test_size=test_size,
                                                       random_state=args.data_random_state)

        # Make sure test_indices is a multiple of batch_size else round down
        len_test_indices = self._round_down(len(test_indices), args.batch_size)
        extra_test = test_indices[len_test_indices:]
        test_indices = test_indices[:len_test_indices]
        train_indices = np.concatenate([train_indices, extra_test])

        self._train_meta = list(np.array(self._metadata)[train_indices])
        self._test_meta = list(np.array(self._metadata)[test_indices])

        self.test_steps = len(self._test_meta) // args.batch_size

        if args.test_size is None:
            assert args.test_batches == self.test_steps

        # pad input sequences with the <pad_token> 0 ( _ )
        self._pad = 0
        self._bos = 102
        # explicitely setting the padding to a value that doesn't originally exist in the spectogram
        # to avoid any possible conflicts, without affecting the output range of the model too much
        if args.symmetric_mels:
            self._target_pad = -args.max_abs_value
        else:
            self._target_pad = 0.
        # Mark finished sequences with 1s
        self._token_pad = 1.
    

    def _get_test_groups(self):
        meta = self._test_meta[self._test_offset]
        self._test_offset += 1

        src_mel = np.load(os.path.join(self._src_mel_dir, 'mel-{}.npy'.format(meta[0])))
        trg_mel = np.load(os.path.join(self._trg_mel_dir, 'mel-{}.npy'.format(meta[0])))
        trg_linear = np.load(os.path.join(self._trg_linear_dir, 'linear-{}.npy'.format(meta[0])))
        # if self.args.source_aux_decoder:
        # src_phoneme = meta[6]
        src_phoneme = np.asarray(phoneme_to_sequence(meta[5], 'en'), dtype=np.int32)
        # if self.args.target_aux_decoder:
        # trg_phoneme = meta[5]
        trg_phoneme = np.asarray(phoneme_to_sequence(meta[5], 'en'), dtype=np.int32)

        token_target = np.asarray([0.] * (len(trg_mel) - 1))

        return (src_mel, src_phoneme, trg_mel, trg_phoneme, trg_linear, token_target)

    def make_train_batches(self):
        # Read a group of examples
        batchset = []
        examples = [self._get_next_example(i) for i in range(len(self._train_meta))]
        examples.sort(key=lambda x: -len(x[0]))
        n = self.args.batch_size * _batches_per_group
        start = 0
        groups = []
        while True:
            end = min(start + n, len(examples))
            group = examples[start: end]
            np.random.shuffle(group)
            groups.append(group)
            if end == len(examples):
                break
            start = end

        for group in groups:
            start = 0
            while True:
                end = min(start + self.args.batch_size, len(group))
                batch = group[start: end]
                batch.sort(key=lambda x: -len(x[0]))
                batchset.append(batch)
                if end == len(group):
                    break
                start = end

        # Bucket examples based on similar output sequence length for efficiency
        np.random.shuffle(batchset)

        return batchset

    def make_test_batches(self):
        start = time.time()

        # Read a group of examples
        n = self.args.batch_size

        # Test on entire test set
        examples = [self._get_test_groups() for i in range(len(self._test_meta))]

        # Bucket examples based on similar output sequence length for efficiency
        examples.sort(key=lambda x: -len(x[0]))
        batches = [examples[i: i + n] for i in range(0, len(examples), n)]
        np.random.shuffle(batches)

        logging.info('\nGenerated {} test batches of size {} in {:.3f} sec'.format(len(batches), n, time.time() - start))
        return batches

    def _get_next_example(self, offset):
        """Gets a single example (input, mel_target, token_target, linear_target, mel_length) from_ disk
        """
        example = self._train_meta[offset]

        src_mel = np.load(os.path.join(self._src_mel_dir, 'mel-{}.npy'.format(example[0])))
        trg_mel = np.load(os.path.join(self._trg_mel_dir, 'mel-{}.npy'.format(example[0])))
        trg_linear = np.load(os.path.join(self._trg_linear_dir, 'linear-{}.npy'.format(example[0])))
        src_phoneme = np.asarray(phoneme_to_sequence(example[5], 'en'), dtype=np.int32)
        trg_phoneme = np.asarray(phoneme_to_sequence(example[5], 'en'), dtype=np.int32)

        # Create parallel sequences containing zeros to represent a non finished sequence
        token_target = np.asarray([0.] * (len(trg_mel) - 1))
        return (src_mel, src_phoneme, trg_mel, trg_phoneme, trg_linear, token_target)

    def _round_down(self, x, multiple):
        remainder = x % multiple
        return x if remainder == 0 else x - remainder
