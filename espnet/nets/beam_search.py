from itertools import chain
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

import torch

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.scorer_interface import PartialScorerInterface
from espnet.nets.scorer_interface import ScorerInterface


class Hypothesis(NamedTuple):
    """Hypothesis data type"""

    yseq: torch.Tensor
    score: float = 0
    scores: Dict[str, float] = dict()
    states: Dict[str, Dict] = dict()

    def asdict(self) -> dict:
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v) for k, v in self.scores.items()}
        )._asdict()


class BeamSearch(torch.nn.Module):
    """Beam search implementation class

    Args:
        scorers (dict[str, ScorerInterface]): Dict of decoder modules e.g., Decoder, CTCPrefixScorer, LM
            The scorer will be ignored if it is `None`
        weights (dict[str, float]): Dict of weights for each scorers
            The scorer will be ignored if its weight is 0
        beam_size (int): The number of hypotheses kept during search
        vocab_size (int): The number of vocabulary
        sos (int): Start of sequence id
        eos (int): End of sequence id
        token_list (list[str]): List of tokens for debug log
        pre_beam_score (str): key of scores to perform pre-beam search
        pre_beam_ratio (float): beam size in the pre-beam search will be `int(pre_beam_ratio * beam_size)`
    """

    def __init__(self, scorers: Dict[str, ScorerInterface], weights: Dict[str, float],
                 beam_size: int, vocab_size: int,
                 sos: int, eos: int, token_list: List[str] = None,
                 pre_beam_ratio: float = 1.5, pre_beam_score: str = "decoder"):
        super().__init__()
        # set scorers
        self.weights = weights
        self.full_scorers = dict()
        self.part_scorers = dict()
        self.nn_dict = torch.nn.ModuleDict()
        for k, v in scorers.items():
            w = weights.get(k, 0)
            if w == 0 or v is None:
                continue
            assert isinstance(v, ScorerInterface), f"{k} ({type(v)}) does not implement ScorerInterface"
            if isinstance(v, PartialScorerInterface):
                self.part_scorers[k] = v
            else:
                self.full_scorers[k] = v
            if isinstance(v, torch.nn.Module):
                self.nn_dict[k] = v

        # set configurations
        self.sos = sos
        self.eos = eos
        self.token_list = token_list
        self.pre_beam_size = int(pre_beam_ratio * beam_size)
        self.beam_size = beam_size
        self.n_vocab = vocab_size
        self.pre_beam_score = pre_beam_score

    def init_hyp(self, x: torch.Tensor) -> Hypothesis:
        init_states = dict()
        init_scores = dict()
        for k, d in chain(self.full_scorers.items(), self.part_scorers.items()):
            init_states[k] = d.init_state(x)
            init_scores[k] = 0.0
        return Hypothesis(
            score=0.0, scores=init_scores, states=init_states,
            yseq=torch.tensor([self.sos], device=x.device))

    @staticmethod
    def append_token(xs: torch.Tensor, x: int) -> torch.Tensor:
        x = torch.tensor([x], dtype=xs.dtype, device=xs.device)
        return torch.cat((xs, x))

    def score(self, hyp: Hypothesis, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        scores = dict()
        states = dict()
        for k, d in self.full_scorers.items():
            scores[k], states[k] = d.score(hyp.yseq, hyp.states[k], x)
        return scores, states

    def score_partial(self, hyp: Hypothesis, ids: torch.Tensor, x: torch.Tensor) \
            -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        scores = dict()
        states = dict()
        for k, d in self.part_scorers.items():
            scores[k], states[k] = d.score_partial(hyp.yseq, ids, hyp.states[k], x)
        return scores, states

    def main_beam(self, weighted_scores: torch.Tensor, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if weighted_scores.size(0) == ids.size(0):
            # no pre beam performed
            top_ids = weighted_scores.topk(self.beam_size)[1]
            return top_ids, top_ids

        # mask pruned in the pre-beam search
        tmp = weighted_scores[ids]
        weighted_scores[:] = -float("inf")
        weighted_scores[ids] = tmp
        # main beam search
        top_ids = weighted_scores.topk(self.beam_size)[1]
        local_ids = weighted_scores[ids].topk(self.beam_size)[1]
        return top_ids, local_ids

    @staticmethod
    def merge_scores(hyp: Hypothesis, scores: Dict[str, torch.Tensor], idx: int,
                     part_scores: Dict[str, torch.Tensor], part_idx: int) -> Dict[str, torch.Tensor]:
        new_scores = dict()
        for k, v in scores.items():
            new_scores[k] = hyp.scores[k] + v[idx]
        for k, v in part_scores.items():
            new_scores[k] = v[part_idx]
        return new_scores

    def merge_states(self, states: Any, part_states: Any, local_j: int) -> Any:
        new_states = dict()
        for k, v in states.items():
            new_states[k] = v
        for k, d in self.part_scorers.items():
            new_states[k] = d.select_state(part_states[k], local_j)
        return new_states

    def forward(self, x: torch.Tensor, maxlenratio: float = 0.0, minlenratio: float = 0.0) -> List[Hypothesis]:
        """Perform beam search

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[Hypothesis]: N-best decoding results
        """
        # set length bounds
        if maxlenratio == 0:
            maxlen = x.shape[0]
        else:
            maxlen = max(1, int(maxlenratio * x.size(0)))
        minlen = int(minlenratio * x.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # main loop of prefix search
        running_hyps = [self.init_hyp(x)]
        ended_hyps = []
        for i in range(maxlen):
            logging.debug('position ' + str(i))
            best = []
            for hyp in running_hyps:
                scores, states = self.score(hyp, x)
                # pre beam
                if self.pre_beam_size < self.n_vocab and self.pre_beam_score in scores:
                    part_ids = torch.topk(scores[self.pre_beam_score], self.pre_beam_size)[1]
                else:
                    part_ids = torch.arange(self.n_vocab, device=x.device)
                part_scores, part_states = self.score_partial(hyp, part_ids, x)

                # main beam
                weighted_scores = torch.zeros(self.n_vocab, dtype=x.dtype, device=x.device)
                for k in self.full_scorers:
                    weighted_scores += self.weights[k] * scores[k]
                for k in self.part_scorers:
                    weighted_scores[part_ids] += self.weights[k] * part_scores[k]
                weighted_scores += hyp.score

                # update hyps
                for j, part_j in zip(*self.main_beam(weighted_scores, part_ids)):
                    # will be (2 x beam at most)
                    best.append(Hypothesis(
                        score=(weighted_scores[j]),
                        yseq=self.append_token(hyp.yseq, j),
                        scores=self.merge_scores(hyp, scores, j, part_scores, part_j),
                        states=self.merge_states(states, part_states, part_j)))

                # sort and prune 2 x beam -> beam
                best = sorted(best, key=lambda x: x.score, reverse=True)[:self.beam_size]

            # post process of one iteration
            running_hyps = self.post_process(i, maxlen, maxlenratio, best, ended_hyps)
            if len(running_hyps) == 0:
                logging.info('no hypothesis. Finish decoding.')
                break

        nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)[:self.beam_size]
        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            return self.forward(x=x, maxlenratio=maxlenratio, minlenratio=max(0.0, minlenratio - 0.1))

        # report the best result
        best = nbest_hyps[0]
        logging.info(f'total log probability: {best.score}')
        logging.info(f'normalized log probability: {best.score / len(best.yseq)}')
        return nbest_hyps

    def post_process(self, i: int, maxlen: int, maxlenratio: float,
                     running_hyps: List[Hypothesis], ended_hyps: List[Hypothesis]) -> List[Hypothesis]:
        logging.debug(f'the number of running hypothes: {len(running_hyps)}')
        if self.token_list is not None:
            logging.debug("best hypo: " + "".join([self.token_list[x] for x in running_hyps[0].yseq[1:]]))
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.info("adding <eos> in the last position in the loop")
            running_hyps = [h._replace(yseq=self.append_token(h.yseq, self.eos)) for h in running_hyps]

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a probmlem, number of hyps < beam)
        remained_hyps = []
        for hyp in running_hyps:
            if hyp.yseq[-1] == self.eos:
                # e.g., Word LM needs to add final <eos> score
                for k, d in chain(self.full_scorers.items(), self.part_scorers.items()):
                    s = d.final_score(hyp.states[k])
                    hyp.scores[k] += s
                    hyp = hyp._replace(score=hyp.score + self.weights[k] * s)
                ended_hyps.append(hyp)
            else:
                remained_hyps.append(hyp)
        # end detection
        if maxlenratio == 0.0 and end_detect([h.asdict() for h in ended_hyps], i):
            logging.info(f'end detected at {i}')
            return []
        if len(remained_hyps) > 0:
            logging.debug(f'remeined hypothes: {len(remained_hyps)}')
        return remained_hyps


def beam_search(x: torch.Tensor, sos: int, eos: int, beam_size: int, vocab_size: int,
                scorers: Dict[str, ScorerInterface], weights: Dict[str, float],
                token_list: List[str] = None, maxlenratio: float = 0.0, minlenratio: float = 0.0,
                pre_beam_ratio: float = 1.5, pre_beam_score: str = "decoder") -> list:
    """Beam search with scorers

    Args:
        x (torch.Tensor): Encoded speech feature (T, D)
        sos (int): Start of sequence id
        eos (int): End of sequence id
        beam_size (int): The number of hypotheses kept during search
        vocab_size (int): The number of vocabulary
        scorers (dict[str, ScorerInterface]): Dict of decoder modules e.g., Decoder, CTCPrefixScorer, LM
            The scorer will be ignored if it is `None`
        weights (dict[str, float]): Dict of weights for each scorers
            The scorer will be ignored if its weight is 0
        token_list (list[str]): List of tokens for debug log
        maxlenratio (float): Input length ratio to obtain max output length.
            If maxlenratio=0.0 (default), it uses a end-detect function
            to automatically find maximum hypothesis lengths
        minlenratio (float): Input length ratio to obtain min output length.
        pre_beam_score (str): key of scores to perform pre-beam search
        pre_beam_ratio (float): beam size in the pre-beam search will be `int(pre_beam_ratio * beam_size)`

    Returns:
        list: N-best decoding results
    """
    ret = BeamSearch(
        scorers, weights,
        beam_size=beam_size,
        vocab_size=vocab_size,
        pre_beam_ratio=pre_beam_ratio,
        pre_beam_score=pre_beam_score,
        sos=sos,
        eos=eos,
        token_list=token_list,
    ).forward(
        x=x,
        maxlenratio=maxlenratio,
        minlenratio=minlenratio)
    return [h.asdict() for h in ret]
