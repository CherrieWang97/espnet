# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
from distutils.util import strtobool

import logging
import math
import pdb

import torch
import chainer

from chainer import reporter
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.encoder_layer import LMPredictionHead
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.gelu_feed_forward import GeluFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer

class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, loss, acc, loss_ctc):
        """Report at every step."""
        reporter.report({'acc': acc}, self)
        reporter.report({'loss': loss}, self)
        reporter.report({'loss_ctc': loss_ctc}, self)


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                           choices=["conv2d", "linear", "embed"],
                           help='transformer input layer type')
        group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                           help='dropout in transformer attention. use --dropout-rate if None is set')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        group.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                           help='normalize loss by length')

        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        # Encoder
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        # Attention
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        return parser

    @property
    def attention_plot_class(self):
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1, asr_model=None, mt_model=None):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.speech_embed = Conv2dSubsampling(idim, args.adim, args.dropout_rate)
        self.word_embed = torch.nn.Embedding(odim, args.adim, padding_idx=0)
        self.position_embed = torch.nn.Embedding(512, args.adim)
        self.lang_embed = torch.nn.Embedding(2, args.adim)
        positionwise_args = (args.adim, args.eunits, args.dropout_rate)
        positionwise_layer = GeluFeedForward
        self.encoder = repeat(
            args.elayers,
            lambda: EncoderLayer(
                args.adim,
                MultiHeadedAttention(args.aheads, args.adim, args.dropout_rate),
                feed_forward=positionwise_layer(*positionwise_args),
                dropout_rate=args.dropout_rate,
                normalize_before=False,
                concat_after=False
             )
        )
        self.embed_norm = LayerNorm(args.adim)
        self.dropout = torch.nn.Dropout(args.dropout_rate)
        self.predict = LMPredictionHead(args.adim, odim)
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = [1]
        self.reporter = Reporter()
        self.acc = torch.zeros(1)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # self.verbose = args.verbose
        self.reset_parameters(args)
        self.tie_weights()
        self.adim = args.adim
        self.ctc = CTC(odim, args.adim, args.dropout_rate, ctc_type='builtin', reduce=True)
        args.report_cer = False
        args.report_wer = False

        from espnet.nets.e2e_asr_common import ErrorCalculator
        self.error_calculator = ErrorCalculator(args.char_list,
                                                args.sym_space, args.sym_blank,
                                                args.report_cer, args.report_wer)
        self.rnnlm = None

    def reset_parameters(self, args):
        # initialize parameters
        initialize(self, args.transformer_init)

    def tie_weights(self):
        self.predict.decoder.weight = self.word_embed.weight

    def load_weight_from_bert(self, path):
        bert_state = torch.load(path)
        self.word_embed.weight.data = bert_state['bert.embeddings.word_embeddings.weight'].data
        self.position_embed.weight.data = bert_state['bert.embeddings.position_embeddings.weight'].data
        self.lang_embed.weight.data = bert_state['bert.embeddings.token_type_embeddings.weight'].data
        self.embed_norm.weight.data = bert_state['bert.embeddings.LayerNorm.weight'].data
        self.embed_norm.bias.data = bert_state['bert.embeddings.LayerNorm.bias'].data
        for i in range(len(self.encoder)):
            self.encoder[i].self_attn.linear_q.weight.data = bert_state['bert.encoder.layer.{}.attention.self.query.weight'.format(i)].data
            self.encoder[i].self_attn.linear_q.bias.data = bert_state['bert.encoder.layer.{}.attention.self.query.bias'.format(i)].data
            self.encoder[i].self_attn.linear_k.weight.data = bert_state['bert.encoder.layer.{}.attention.self.key.weight'.format(i)].data
            self.encoder[i].self_attn.linear_k.bias.data = bert_state['bert.encoder.layer.{}.attention.self.key.bias'.format(i)].data
            self.encoder[i].self_attn.linear_v.weight.data = bert_state['bert.encoder.layer.{}.attention.self.value.weight'.format(i)].data
            self.encoder[i].self_attn.linear_v.bias.data = bert_state['bert.encoder.layer.{}.attention.self.value.bias'.format(i)].data
            self.encoder[i].self_attn.linear_out.weight.data = bert_state['bert.encoder.layer.{}.attention.output.dense.weight'.format(i)].data
            self.encoder[i].self_attn.linear_out.bias.data = bert_state['bert.encoder.layer.{}.attention.output.dense.bias'.format(i)].data
            self.encoder[i].feed_forward.w_1.weight.data = bert_state['bert.encoder.layer.{}.intermediate.dense.weight'.format(i)].data
            self.encoder[i].feed_forward.w_1.bias.data = bert_state['bert.encoder.layer.{}.intermediate.dense.bias'.format(i)].data
            self.encoder[i].feed_forward.w_2.weight.data = bert_state['bert.encoder.layer.{}.output.dense.weight'.format(i)].data
            self.encoder[i].feed_forward.w_2.bias.data = bert_state['bert.encoder.layer.{}.output.dense.bias'.format(i)].data
            self.encoder[i].norm1.weight.data = bert_state['bert.encoder.layer.{}.attention.output.LayerNorm.weight'.format(i)].data
            self.encoder[i].norm1.bias.data = bert_state['bert.encoder.layer.{}.attention.output.LayerNorm.bias'.format(i)].data
            self.encoder[i].norm2.weight.data = bert_state['bert.encoder.layer.{}.output.LayerNorm.weight'.format(i)].data
            self.encoder[i].norm2.bias.data = bert_state['bert.encoder.layer.{}.output.LayerNorm.bias'.format(i)].data
        self.predict.dense.weight.data = bert_state['cls.predictions.transform.dense.weight'].data
        self.predict.dense.bias.data = bert_state['cls.predictions.transform.dense.bias'].data
        self.predict.norm.weight.data = bert_state['cls.predictions.transform.LayerNorm.weight'].data
        self.predict.norm.bias.data = bert_state['cls.predictions.transform.LayerNorm.bias'].data
        self.predict.decoder.weight.data = bert_state['cls.predictions.decoder.weight'].data
        self.predict.bias.data = bert_state['cls.predictions.bias'].data
            

    def forward(self, xs_pad, ilens, ys_pad_src, ys_lens, ys_pad_trg, ys_pad=None, task="tlm"):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        if task == "mlm" or task == "cmlm":
            ys_pad_src = ys_pad_src[:, :max(ys_lens)]
            ys_pad_trg = ys_pad_trg[:, :max(ys_lens)]
            lang_ids = torch.zeros_like(ys_pad_src).long().to(ys_pad_src.device)
            #lang_mask = ys_pad_src == 2
            src_mask = (~make_pad_mask(ys_lens.tolist())).to(ys_pad_src.device).unsqueeze(-2)
            #lang_ids = lang_ids.masked_fill(lang_mask, 1).long()
            #ys_pad_trg = ys_pad_trg.masked_fill(~lang_mask, -1)
            ys_embeded = self.word_embed(ys_pad_src)
            seq_length = ys_pad_src.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=ys_pad_src.device)
            position_ids = position_ids.unsqueeze(0).expand_as(ys_pad_src)
            position_embeded = self.position_embed(position_ids)
            lang_ids = torch.zeros_like(ys_pad_src)
            lang_embeded = self.lang_embed(lang_ids)
            embeddings = ys_embeded + position_embeded + lang_embeded
            embeddings = self.dropout(self.embed_norm(embeddings))
          
            hs_pad, hs_mask = self.encoder(embeddings, src_mask)
            pred_pad = self.predict(hs_pad)
            loss = self.criterion(pred_pad.view(-1, self.odim), ys_pad_trg.contiguous().view(-1))
            acc = th_accuracy(pred_pad.view(-1, self.odim), ys_pad_trg,
                              ignore_label=self.ignore_id)

        if task == "tlm":
            #Conv layer for speech input
            #pdb.set_trace()
            xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
            src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
            speech_pad, speech_mask = self.speech_embed(xs_pad, src_mask)
            lang_ids = torch.ones((speech_pad.size(0), speech_pad.size(1))).long().to(xs_pad.device)
            speech_embeddings = speech_pad + self.lang_embed(lang_ids)
            #Text embedding
            ys_pad_src = ys_pad_src[:, :max(ys_lens)]
            ys_pad_trg = ys_pad_trg[:, :max(ys_lens)]
            lang_ids = torch.zeros_like(ys_pad_src).long().to(ys_pad_src.device)
            text_mask = (~make_pad_mask(ys_lens.tolist())).to(ys_pad_src.device).unsqueeze(-2)
            ys_embeded = self.word_embed(ys_pad_src)
            seq_length = ys_pad_src.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=ys_pad_src.device)
            position_ids = position_ids.unsqueeze(0).expand_as(ys_pad_src)
            position_embeded = self.position_embed(position_ids)
            lang_ids = torch.zeros_like(ys_pad_src)
            lang_embeded = self.lang_embed(lang_ids)
            word_embeddings = ys_embeded + position_embeded + lang_embeded
            #Concat speech embedding and word embedding
            embeddings = torch.cat([word_embeddings, speech_embeddings], dim=1)
            embeddings = self.dropout(self.embed_norm(embeddings))
            mask = torch.cat([text_mask, speech_mask], dim=-1)
            # encoder forward
            hs_pad, hs_mask = self.encoder(embeddings, mask)
            # masked language model predict
            hs_pred = hs_pad[:, :max(ys_lens)]
            pred_pad = self.predict(hs_pred)
            loss = self.criterion(pred_pad.view(-1, self.odim), ys_pad_trg.contiguous().view(-1))
            acc = th_accuracy(pred_pad.view(-1, self.odim), ys_pad_trg,
                              ignore_label=self.ignore_id)
            # CTC predict
            speech_pred = hs_pad[:, max(ys_lens):]
            batch_size = hs_pad.size(0)
            hlens = speech_mask.view(batch_size, -1).sum(1)
            ctc_loss = self.ctc(speech_pred.view(batch_size, -1, self.adim), hlens, ys_pad)

        self.loss = loss + ctc_loss
        loss_data = float(loss)
        ctc_loss_data = float(ctc_loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_data, acc, ctc_loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, feat):
        """Encode acoustic features."""
        self.eval()
        feat = torch.as_tensor(feat).cuda().unsqueeze(0)
        enc_output, _ = self.encoder(feat, None)
        return enc_output.squeeze(0)

    def recognize(self, feat, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """recognize feat.

        :param ndnarray x: input acouctic feature (B, T, D) or (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list

        TODO(karita): do not recompute previous attention for faster decoding
        """
        enc_output = self.encode(feat).unsqueeze(0)
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)

        logging.info('input lengths: ' + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y]}
        if lpz is not None:
            import numpy

            from espnet.nets.ctc_prefix_score import CTCPrefixScore

            ctc_prefix_score = CTCPrefixScore(lpz.cpu().detach().numpy(), 0, self.eos, numpy)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six
        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1, device='cuda').unsqueeze(0)
                ys = torch.tensor(hyp['yseq']).cuda().unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(self.decoder.recognize, (ys, ys_mask, enc_output))
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)
                else:
                    local_att_scores = self.decoder.recognize(ys, ys_mask, enc_output)

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0].cpu(), hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]].cpu() \
                        + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]].cpu()
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            from espnet.nets.e2e_asr_common import end_detect
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(feat, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_asr=None):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                ret[name] = m.attn.cpu().numpy()
        return ret
