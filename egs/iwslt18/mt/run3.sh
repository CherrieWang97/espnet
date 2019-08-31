#!/bin/bash

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch # chainer or pytorch
stage=-1        # start from -1 if you need to start from data download
stop_stage=100
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump    # directory to dump full features
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume= #/teamscratch/tts_intern_experiment/v-chengw/iwslt18/exp4mt/seq2seq_char/results/snapshot.ep.5000       # Resume the training from snapshot
seed=1          # seed to generate random number
# feature configuration
do_delta=false

train_config=conf/train.yaml
decode_config=conf/decode.yaml

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# preprocessing related
src_case=lc.rm
tgt_case=lc
nbpe=5000
bpemode=unigram
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.
datadir=/teamscratch/tts_intern_experiment/v-chengw/iwslt18/data4mt
# data4mt
#  |_ train/
#  |_ other/
#  |_ dev/
#  |_ test/
# Download data from here:

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

. ./path.sh
. ./cmd.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodev
train_set_prefix=train_nodev
train_dev=train_dev
recog_set="dev"


dict=/teamscratch/tts_intern_experiment/v-chengw/iwslt18/data4mt/dict/ted_en.txt
# NOTE: skip stage 3: LM Preparation

expname=seq2seq_pred_subword
expdir=/teamscratch/tts_intern_experiment/v-chengw/iwslt18/exp4mt/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    mkdir -p ${expdir}
    mkdir -p exp/${expname}

    ${cuda_cmd} --gpu ${ngpu} exp/${expname}/train.log \
        mt_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict-tgt ${dict} \
        --tgt-vocab 5001 \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-src /teamscratch/tts_intern_experiment/v-chengw/iwslt18/data4asr/newdata/train/tokenid_subword.txt \
        --train-trg /teamscratch/tts_intern_experiment/v-chengw/iwslt18/exp4st/asr_ctc2/decode_newdump/data_clean.txt \
        --repeat /teamscratch/tts_intern_experiment/v-chengw/iwslt18/exp4st/asr_ctc2/decode_newdump/repeat.txt
        #--valid-src /hdfs/resrchvc/v-chengw/iwslt18/data4mt/st/dev/text.en.share.id \
        #--valid-trg /hdfs/resrchvc/v-chengw/iwslt18/data4mt/st/dev/text.de.share.id
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=16

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_debug
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        mkdir -p ${expdir}/${decode_dir}
        mkdir -p exp/${decode_dir}
        # split data

        #### use CPU for decoding
        ngpu=1

        ${decode_cmd} exp/${decode_dir}/decode3.log \
            CUDA_VISIBLE_DEVICES=3 mt_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-path /teamscratch/tts_intern_experiment/v-chengw/iwslt18/data4mt/allTed/train/split_train03 \
            --result-label ${expdir}/${decode_dir}/result03 \
            --model ${expdir}/results/snapshot.ep.15000

        #score_bleu.sh --case ${tgt_case} --nlsyms ${nlsyms} ${expdir}/${decode_dir} fr ${dict_tgt} ${dict_src}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
