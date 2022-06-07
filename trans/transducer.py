"""Defines a neural transducer."""
from typing import Any, Dict, List, Optional, Tuple
import dataclasses
import functools
import heapq

import torch
import numpy as np

from trans import optimal_expert
from trans import vocabulary
from trans.actions import ConditionalCopy, ConditionalDel, ConditionalIns, \
    ConditionalSub, Edit, EndOfSequence, GenerativeEdit, BeginOfSequence
from trans.vocabulary import BEGIN_WORD, COPY, DELETE, END_WORD


MAX_ACTION_SEQ_LEN = 150


@functools.total_ordering
@dataclasses.dataclass
class Output:
    action_history: List[Any]
    output: str
    log_p: float
    losses: List[torch.Tensor] = None

    def __lt__(self, other):
        return self.log_p < other.log_p

    def __eq__(self, other):
        return self.log_p == other.log_p


@dataclasses.dataclass
class Hypothesis:
    action_history: List[Any]
    alignment: int
    decoder: Tuple[torch.Tensor, torch.Tensor]
    negative_log_p: float
    output: List[str]


@functools.total_ordering
@dataclasses.dataclass
class Expansion:
    action: Any
    decoder: Tuple[torch.Tensor, torch.Tensor]
    from_hypothesis: Hypothesis
    negative_log_p: float

    def __lt__(self, other):
        return self.negative_log_p < other.negative_log_p

    def __eq__(self, other):
        return self.negative_log_p == other.negative_log_p


class Transducer(torch.nn.Module):
    def __init__(self, vocab: vocabulary.Vocabularies,
                 expert: optimal_expert.Expert, char_dim: int, action_dim: int,
                 enc_hidden_dim: int, enc_layers: int, enc_dropout: float,
                 dec_hidden_dim: int, dec_layers: int, device: str = 'cpu', **kwargs):

        super().__init__()
        self.device = torch.device(device)

        self.vocab = vocab
        self.optimal_expert = expert

        self.number_characters = len(vocab.characters)
        self.number_actions = len(vocab.actions)
        self.substitutions = self.vocab.substitutions
        self.inserts = self.vocab.insertions

        # encoder
        self.char_lookup = torch.nn.Embedding(
            num_embeddings=self.number_characters,
            embedding_dim=char_dim,
            device=self.device,
        )

        self.enc = torch.nn.LSTM(
            input_size=char_dim,
            hidden_size=enc_hidden_dim,
            num_layers=enc_layers,
            dropout=enc_dropout,
            bidirectional=True,
            device=self.device,
        )

        # decoder
        self.act_lookup = torch.nn.Embedding(
            num_embeddings=self.number_actions,
            embedding_dim=action_dim,
            device=self.device,
        )

        decoder_input_dim = enc_hidden_dim * 2 + action_dim

        self.dec = torch.nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=dec_hidden_dim,
            num_layers=dec_layers,
            device=self.device,
        )

        self.h0_c0 = (  # batch size 1
            torch.zeros((dec_layers, 1, dec_hidden_dim), device=self.device),
            torch.zeros((dec_layers, 1, dec_hidden_dim), device=self.device),
        )

        # classifier
        self.W = torch.nn.Linear(
            in_features=dec_hidden_dim,
            out_features=self.number_actions,
            device=self.device,

        )

    def input_embedding(self, input_: List[int], is_training: bool):
        """Returns a list of character embeddings for the input."""
        input_tensor = torch.tensor(
            input_,
            dtype=torch.int,
            device=self.device,
        )
        emb = self.char_lookup(input_tensor)

        if not is_training:

            unk_indices = [
                j for j, i in enumerate(input_) if i >= self.number_characters]

            if unk_indices:
                # UNK is the average of trained embeddings (excluding UNK)
                ids_tensor = torch.tensor(
                    range(1, self.number_characters),
                    dtype=torch.int,
                    device=self.device,
                )
                unk = self.char_lookup(ids_tensor).mean(dim=0)
                emb[unk_indices] = unk

        return emb.unsqueeze(1)  # Adds batch dimension.

    def compute_valid_actions(self, length_encoder_suffix: int) -> List[int]:
        valid_actions = [END_WORD]
        valid_actions.extend(self.inserts)
        if length_encoder_suffix > 1:
            valid_actions.extend([COPY, DELETE])
            valid_actions.extend(self.substitutions)
        return valid_actions

    @staticmethod
    def sample(log_probs: np.array) -> int:
        """Samples an action from a log-probability distribution."""
        dist = np.exp(log_probs)
        rand = np.random.rand()
        for action, p in enumerate(dist):
            rand -= p
            if rand <= 0:
                break
        return action

    @staticmethod
    def remap_actions(action_scores: Dict[Any, float]) -> Dict[Any, float]:
        """Maps a generative oracle's edit to their conditional counterparts."""
        remapped_action_scores = dict()
        for action, score in action_scores.items():
            if isinstance(action, GenerativeEdit):
                remapped_action = action.conditional_counterpart()
            elif isinstance(action, Edit):
                remapped_action = action
            else:
                raise ValueError(f"Unknown action: {action, score}.\n"
                                 f"action_scores: {action_scores}")
            remapped_action_scores[remapped_action] = score
        return remapped_action_scores

    def expert_rollout(self, input_: str, target: str, alignment: int,
                       prediction: List[str]) -> List[int]:
        """Rolls out with optimal expert policy.

        Args:
            input_: Input string (x).
            target: Target prediction (t).
            alignment: Position of control in the input string.
            prediction: The current prediction so far (y).

        Returns:
            List of optimal actions as integer codes."""
        raw_action_scores = self.optimal_expert.score(
            input_, target, alignment, prediction)
        action_scores = self.remap_actions(raw_action_scores)

        optimal_value = min(action_scores.values())
        return [self.vocab.encode_unseen_action(action)
                for action, value in action_scores.items()
                if value == optimal_value]

    def mark_as_invalid(self, logits: torch.Tensor,
                        valid_actions: List[int]) -> torch.Tensor:
        log_validity = torch.full(
            (self.number_actions,),
            -np.inf,
            device=self.device,
        )  # All actions invalid by default.
        log_validity[valid_actions] = 0.
        return logits + log_validity

    def log_softmax(self, logits: torch.Tensor,
                    valid_actions: List[int]) -> torch.Tensor:
        logits_valid = self.mark_as_invalid(logits, valid_actions)
        return torch.nn.functional.log_softmax(logits_valid, dim=1)

    def log_sum_softmax_loss(self, optimal_actions: List[int],
                             logits: torch.Tensor,
                             valid_actions: List[int]) -> torch.Tensor:
        """Compute log loss similar to Riezler et al 2000."""
        logits_valid = self.mark_as_invalid(logits, valid_actions)
        log_sum_selected_terms = torch.logsumexp(
            logits_valid[:, optimal_actions],
            dim=0,
        )
        enumerated_dim = tuple(range(logits_valid.dim()))
        normalization_term = torch.logsumexp(
            logits_valid,
            dim=enumerated_dim,
        )
        return log_sum_selected_terms - normalization_term

    def transduce(self, input_: str, encoded_input: List[int],
                  target: Optional[str] = None, rollin: Optional[float] = None):
        """Runs the transducer for dynamic-oracle training and greedy decoding.

        Args:
            input_: Input string.
            encoded_input: List of integer character codes.
            target: Target string during training, `None` during prediction.
            rollin: The probability with which an action sampled from the model
                    is executed. Used during training."""
        is_training = bool(target)
        input_emb = self.input_embedding(encoded_input, is_training)
        bidirectional_emb_, _ = self.enc(input_emb)  # L x 1 x E
        bidirectional_emb = bidirectional_emb_[1:]  # drop BEGIN_WORD
        input_length = len(bidirectional_emb)
        decoder = self.h0_c0

        alignment = 0
        action_history: List[torch.IntTensor] = [torch.IntTensor([BEGIN_WORD])]
        output: List[str] = []
        losses: List[torch.Tensor] = []
        log_p = 0.

        while len(action_history) <= MAX_ACTION_SEQ_LEN:

            length_encoder_suffix = input_length - alignment
            valid_actions = self.compute_valid_actions(length_encoder_suffix)

            input_char_embedding = bidirectional_emb[alignment]
            previous_action_embedding = self.act_lookup(torch.tensor([action_history[-1]],
                                                                     device=self.device))
            decoder_input = torch.cat(
                (input_char_embedding, previous_action_embedding),
                dim=1,
            ).unsqueeze(1)  # Adds batch dimension.
            decoder_output_, decoder = self.dec(decoder_input, decoder)

            decoder_output = decoder_output_.squeeze(1)
            logits = self.W(decoder_output)
            log_probs = self.log_softmax(logits, valid_actions)

            log_probs_np = log_probs.squeeze().cpu().detach().numpy()

            if target is None:
                # argmax decoding
                action = np.argmax(log_probs_np)
            else:
                # training with dynamic oracle

                # 1. ACTIONS TO MAXIMIZE
                optim_actions = self.expert_rollout(
                    input_, target, alignment, output)

                loss = self.log_sum_softmax_loss(
                    optim_actions, logits, valid_actions)

                # 2. ACTION SPACE EXPLORATION: NEXT ACTION
                if np.random.rand() <= rollin:
                    # action is picked by sampling
                    action = self.sample(log_probs_np)
                else:
                    # action is picked from optim_actions
                    # reinforce model beliefs by picking highest probability
                    # action that is consistent with oracle
                    action = optim_actions[
                        int(np.argmax([log_probs_np[a] for a in optim_actions]))
                    ]
                losses.append(loss)

            log_p += log_probs_np[action]
            action_history.append(action)
            # execute the action to update the transducer state
            action = self.vocab.decode_action(action)

            if isinstance(action, ConditionalCopy):
                char_ = input_[alignment]
                alignment += 1
                output.append(char_)
            elif isinstance(action, ConditionalDel):
                alignment += 1
            elif isinstance(action, ConditionalIns):
                output.append(action.new)
            elif isinstance(action, ConditionalSub):
                alignment += 1
                output.append(action.new)
            elif isinstance(action, EndOfSequence):
                break
            elif isinstance(action, BeginOfSequence):
                continue
            else:
                raise ValueError(f"Unknown action: {action}.")

        return Output(action_history, "".join(output), log_p, losses)

    def beam_search_decode(self, input_: str, encoded_input: List[int],
                           beam_width: int):
        """Runs the transducer with beam search.

        Args:
            input_: Input string.
            encoded_input: List of integer character codes.
            beam_width: Width of the beam search.
        """
        input_emb = self.input_embedding(encoded_input, is_training=False)
        bidirectional_emb_, _ = self.enc(input_emb)
        bidirectional_emb = bidirectional_emb_[1:]
        input_length = len(bidirectional_emb)
        decoder = self.h0_c0

        beam: List[Hypothesis] = [
            Hypothesis(action_history=[BEGIN_WORD],
                       alignment=0,
                       decoder=decoder,
                       negative_log_p=0.,
                       output=[])]

        hypothesis_length = 0
        complete_hypotheses = []

        while beam and beam_width > 0 and hypothesis_length <= MAX_ACTION_SEQ_LEN:

            expansions: List[Expansion] = []

            for hypothesis in beam:

                length_encoder_suffix = input_length - hypothesis.alignment
                valid_actions = self.compute_valid_actions(length_encoder_suffix)
                # decoder
                decoder_input = torch.cat([
                    bidirectional_emb[hypothesis.alignment],
                    self.act_lookup(torch.tensor([hypothesis.action_history[-1]],
                                                 device=self.device))
                ], dim=1,
                ).unsqueeze(1)
                decoder_output_, decoder = self.dec(decoder_input, hypothesis.decoder)

                decoder_output = decoder_output_.squeeze(1)
                logits = self.W(decoder_output)
                log_probs = self.log_softmax(logits, valid_actions)

                for action in valid_actions:

                    log_p = hypothesis.negative_log_p - \
                            log_probs[0, action]  # min heap, so minus

                    heapq.heappush(expansions,
                                   Expansion(action, decoder,
                                             hypothesis, log_p))

            beam: List[Hypothesis] = []

            for _ in range(beam_width):

                expansion: Expansion = heapq.heappop(expansions)
                from_hypothesis = expansion.from_hypothesis
                action = expansion.action
                action_history = list(from_hypothesis.action_history)
                action_history.append(action)
                output = list(from_hypothesis.output)

                # execute the action to update the transducer state
                action = self.vocab.decode_action(action)

                if isinstance(action, EndOfSequence):
                    # 1. COMPLETE HYPOTHESIS, REDUCE BEAM
                    complete_hypothesis = Output(
                        action_history=action_history,
                        output="".join(output),
                        log_p=-expansion.negative_log_p)  # undo min heap minus

                    complete_hypotheses.append(complete_hypothesis)
                    beam_width -= 1
                else:
                    # 2. EXECUTE ACTION AND ADD FULL HYPOTHESIS TO NEW BEAM
                    alignment = from_hypothesis.alignment

                    if isinstance(action, ConditionalCopy):
                        char_ = input_[alignment]
                        alignment += 1
                        output.append(char_)
                    elif isinstance(action, ConditionalDel):
                        alignment += 1
                    elif isinstance(action, ConditionalIns):
                        output.append(action.new)
                    elif isinstance(action, ConditionalSub):
                        alignment += 1
                        output.append(action.new)
                    else:
                        raise ValueError(f"Unknown action: {action}.")

                    hypothesis = Hypothesis(
                        action_history=action_history,
                        alignment=alignment,
                        decoder=expansion.decoder,
                        negative_log_p=expansion.negative_log_p,
                        output=output)

                    beam.append(hypothesis)

            hypothesis_length += 1

        if not complete_hypotheses:
            # nothing found because the model is very bad
            for hypothesis in beam:

                complete_hypothesis = Output(
                    action_history=hypothesis.action_history,
                    output="".join(hypothesis.output),
                    log_p=-hypothesis.negative_log_p)  # undo min heap minus

                complete_hypotheses.append(complete_hypothesis)

        complete_hypotheses.sort(reverse=True)
        return complete_hypotheses
