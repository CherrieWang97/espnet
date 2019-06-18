#!/usr/bin/env python

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import pdb
import argparse
import logging
import os
import platform
import random
import subprocess
import sys

from distutils.util import strtobool

import numpy as np


def main(args):
    parser = argparse.ArgumentParser()
    # general configuration
    parser.add_argument('--ngpu', default=0, type=int,
                        help='Number of GPUs')
    parser.add_argument('--outdir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--debugmode', default=1, type=int,
                        help='Debugmode')
    parser.add_argument('--seed', default=1, type=int,
                        help='Random seed')
    parser.add_argument('--resume', '-r', default='', type=str, nargs='?',
                        help='Resume the training from snapshot')
    parser.add_argument('--minibatches', '-N', type=int, default='-1',
                        help='Process only N minibatches (for debug)')
    parser.add_argument('--verbose', '-V', default=1, type=int,
                        help='Verbose option')
    parser.add_argument('--tensorboard-dir', default=None, type=str, nargs='?', help="Tensorboard log dir path")
    # task related

    ###add from tf project
    parser.add_argument('--base_dir', default='/home/v-chengw/data/peppapig/')
    parser.add_argument('--source', default='chs')
    parser.add_argument('--target', default='en')
    parser.add_argument('--input', default='train.txt')

    parser.add_argument('--num_mels', type=int, default=80,
                        help="Number of mel-spectrogram channels and local conditioning dimensionality")
    parser.add_argument('--num_freq', type=int, default=513,
                        help="(= n_fft / 2 + 1) only used wchen adding linear spectrograms post processing network")
    parser.add_argument('--hop_size', type=int, default=200,
                        help="0.0125 * sample_rate)")
    parser.add_argument('--win_size', type=int, default=800,
                        help="0.05 * sample_rate")
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--test_size', type=float, default=0.05,
                        help="percent of data to keep as test data")
    parser.add_argument('--test_batches', type=int, help="test batches")

    parser.add_argument('--symmetric_mels', type=strtobool, default=True)
    parser.add_argument('--max_abs_value', type=float, default=4.,
                        help="max absolute value of data. If symmetric, data will be [-max, max] else [0, max]")
    parser.add_argument('--data_random_state', type=int, default=1234)

    # network architecture
    # encoder
    parser.add_argument('--embed_dim', default=512, type=int,
                        help='Number of dimension of embedding')
    parser.add_argument('--elayers', default=4, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--eunits', '-u', default=1024, type=int,
                        help='Number of encoder hidden units')
    # attention
    parser.add_argument('--atype', default="MultiHeadAdd", type=str,
                        choices=["MultiHeadAdd", "Add", "location"],
                        help='Type of attention mechanism')
    parser.add_argument('--adim', default=512, type=int,
                        help='Number of attention transformation dimensions')
    parser.add_argument('--aheads', default=4, type=int,
                        help='Number of attention heads in MultiHead Attention')
    # decoder
    parser.add_argument('--dlayers', default=4, type=int,
                        help='Number of decoder layers')
    parser.add_argument('--dunits', default=1024, type=int,
                        help='Number of decoder hidden units')
    parser.add_argument('--use_srcaux_decoder', default=False, type=strtobool,
                        help='Whether to use source auxiliary decoder')
    parser.add_argument('--use_trgaux_decoder', default=True, type=strtobool,
                        help='Whether to use target auxiliary decoder')
    parser.add_argument('--src_vocab_size', default=120, type=int,
                        help="source phoneme vocabulary size")
    parser.add_argument('--trg_vocab_size', default=120, type=int,
                        help="target phoneme vocabulary size")
    parser.add_argument('--auxlayers', default=2, type=int,
                        help="Number of auxiliary decoder layers")
    parser.add_argument('--auxunits', default=256, type=int,
                        help="Number of auxiliary decoder units")
    parser.add_argument('--src_ilayer', default=4, type=int,
                        help="Intermediate encoder layer passed to source auxiliary decoder")
    parser.add_argument('--trg_ilayer', default=4, type=int,
                        help="Intermediate encoder layer passed to target auxiliary decoder")
    parser.add_argument('--prenet_layers', default=2, type=int,
                        help='Number of prenet layers')
    parser.add_argument('--prenet_units', default=32, type=int,
                        help='Number of prenet hidden units')
    parser.add_argument('--postnet_layers', default=5, type=int,
                        help='Number of postnet layers')
    parser.add_argument('--postnet_chans', default=512, type=int,
                        help='Number of postnet channels')
    parser.add_argument('--postnet_filts', default=5, type=int,
                        help='Filter size of postnet')
    parser.add_argument('--output_activation', default=None, type=str, nargs='?',
                        help='Output activation function')
    # cbhg
    parser.add_argument('--use_cbhg', default=False, type=strtobool,
                        help='Whether to use CBHG module')
    parser.add_argument('--cbhg_conv_bank_layers', default=8, type=int,
                        help='Number of convoluional bank layers in CBHG')
    parser.add_argument('--cbhg_conv_bank_chans', default=128, type=int,
                        help='Number of convoluional bank channles in CBHG')
    parser.add_argument('--cbhg_conv_proj_filts', default=3, type=int,
                        help='Filter size of convoluional projection layer in CBHG')
    parser.add_argument('--cbhg_conv_proj_chans', default=256, type=int,
                        help='Number of convoluional projection channels in CBHG')
    parser.add_argument('--cbhg_highway_layers', default=4, type=int,
                        help='Number of highway layers in CBHG')
    parser.add_argument('--cbhg_highway_units', default=128, type=int,
                        help='Number of highway units in CBHG')
    parser.add_argument('--cbhg_gru_units', default=256, type=int,
                        help='Number of GRU units in CBHG')
    # model (parameter) related
    parser.add_argument('--use_speaker_embedding', default=False, type=strtobool,
                        help='Whether to use speaker embedding')
    parser.add_argument('--use_batch_norm', default=True, type=strtobool,
                        help='Whether to use batch normalization')
    parser.add_argument('--use_concate', default=True, type=strtobool,
                        help='Whether to concatenate encoder embedding with decoder outputs')
    parser.add_argument('--dropout', default=0.2, type=float,
                        help='Dropout rate')
    parser.add_argument('--zoneout', default=0.1, type=float,
                        help='Zoneout rate')
    parser.add_argument('--reduction_factor', default=2, type=int,
                        help='Reduction factor')
    # loss related
    parser.add_argument('--use_masking', default=False, type=strtobool,
                        help='Whether to use masking in calculation of loss')
    parser.add_argument('--bce_pos_weight', default=20.0, type=float,
                        help='Positive sample weight in BCE calculation (only for use_masking=True)')
    # minibatch related
    parser.add_argument('--sortagrad', default=0, type=int, nargs='?',
                        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs")
    parser.add_argument('--batch_sort_key', default='shuffle', type=str,
                        choices=['shuffle', 'output', 'input'], nargs='?',
                        help='Batch sorting key')
    parser.add_argument('--batch-size', '-b', default=8, type=int,
                        help='Batch size')
    parser.add_argument('--maxlen-in', default=100, type=int, metavar='ML',
                        help='Batch size is reduced if the input sequence length > ML')
    parser.add_argument('--maxlen-out', default=200, type=int, metavar='ML',
                        help='Batch size is reduced if the output sequence length > ML')
    parser.add_argument('--n_iter_processes', default=0, type=int,
                        help='Number of processes of iterator')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')
    # optimization related
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate for optimizer')
    parser.add_argument('--eps', default=1e-6, type=float,
                        help='Epsilon for optimizer')
    parser.add_argument('--weight-decay', default=1e-6, type=float,
                        help='Weight decay coefficient for optimizer')
    parser.add_argument('--epochs', '-e', default=30, type=int,
                        help='Number of maximum epochs')
    parser.add_argument('--early-stop-criterion', default='validation/main/loss', type=str, nargs='?',
                        help="Value to monitor to trigger an early stopping of the training")
    parser.add_argument('--patience', default=3, type=int, nargs='?',
                        help="Number of epochs to wait without improvement before stopping the training")
    parser.add_argument('--grad-clip', default=1, type=float,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--num-save-attention', default=5, type=int,
                        help='Number of samples of attention to be saved')
    args = parser.parse_args(args)

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
        logging.warning('Skip DEBUG/INFO messages')

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

    # set random seed
    logging.info('random seed = %d' % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    from espnet.s2s.s2s import train
    train(args)



if __name__ == "__main__":
    main(sys.argv[1:])
