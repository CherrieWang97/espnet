#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh

# general configuration
backend=pytorch # chainer or pytorch
stage=-1        # start from -1 if you need to start from data download
stop_stage=100
ngpu=1          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
N=0             # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0       # verbose option
resume= #/teamscratch/tts_intern_experiment/v-chengw/iwslt18/exp4st/asr_ctc/results/snapshot.ep.25000         # Resume the training from snapshot
seed=1          # seed to generate random number
# feature configuration
do_delta=false

train_config=conf/train.yaml
decode_config=conf/decode.yaml

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# pre-training related
asr_model= #/teamscratch/tts_intern_experiment/v-chengw/iwslt18/exp4st/asr_ctc_char/results/model.acc.best
mt_model= #/teamscratch/tts_intern_experiment/v-chengw/iwslt18/exp4st/mt_char2char_finetune/results/snapshot.ep.50000

# preprocessing related
case=lc
# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
st_ted=/teamscratch/tts_intern_experiment/v-chengw/iwslt18/data
dumpdir=/teamscratch/tts_intern_experiment/v-chengw/iwslt18/data4asr    # directory to dump full features
# st_ted=/n/sd3/inaguma/corpus/iwslt18/data

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

train_set=train_nodevtest_sp.de
train_set_prefix=train_nodevtest_sp
train_dev=dev.de
recog_set="newdump"


feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

dict=/teamscratch/tts_intern_experiment/v-chengw/iwslt18/data4mt/dict/ted_char.txt

# NOTE: skip stage 3: LM Preparation

expname=asr_ctc_char

expdir=/teamscratch/tts_intern_experiment/v-chengw/iwslt18/exp4st/${expname}
mkdir -p ${expdir}
mkdir -p exp
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} exp/${expname}.log \
        multi_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --seed ${seed} \
        --verbose ${verbose} \
        --resume ${resume} \
        --src_vocab 115 \
        --trg_vocab 140 \
        --train-json /teamscratch/tts_intern_experiment/v-chengw/iwslt18/data4st/train/data_newchar.json \
        --valid-json /teamscratch/tts_intern_experiment/v-chengw/iwslt18/data4st/dev.de/deltafalse/data_newchar.json \
        --asr-json /teamscratch/tts_intern_experiment/v-chengw/iwslt18/data4asr/train/deltafalse/data_newchar.json \
        --train-src /teamscratch/tts_intern_experiment/v-chengw/iwslt18/data4mt/allTed/train/train.en.char.id \
        --train-trg /teamscratch/tts_intern_experiment/v-chengw/iwslt18/data4mt/allTed/train/train.de.char.id 
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}
        feat_recog_dir=${dumpdir}/${rtask}

        # split data
        #splitjson.py --parts ${nj} ${feat_recog_dir}/data.${case}.json

        #### use CPU for decoding
        ngpu=1
        mkdir -p exp/${decode_dir}
        mkdir -p ${expdir}/${decode_dir}

        ${decode_cmd} JOB=12 exp/${decode_dir}/log/decode.JOB.log \
            CUDA_VISIBLE_DEVICES=1 multi_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_uniq.JOB.json \
            --result-label ${expdir}/${decode_dir}/data_uniq.JOB \
            --model ${expdir}/results/${recog_model} \
            --char-list /teamscratch/tts_intern_experiment/v-chengw/iwslt18/data4mt/dict/dict_char.txt

        if [ ${rtask} = "dev.de" ] || [ ${rtask} = "test.de" ]; then
            score_bleu.sh --case ${case} --nlsyms ${nlsyms} ${expdir}/${decode_dir} de ${dict}
        else
            set=$(echo ${rtask} | cut -f -1 -d ".")
            local/score_bleu_reseg.sh --bpemodel /teamscratch/tts_intern_experiment/v-chengw/iwslt18/data4mt/dict/ted_de.model ${expdir}/${decode_dir} ${dict} ${st_ted} ${set}
        fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
