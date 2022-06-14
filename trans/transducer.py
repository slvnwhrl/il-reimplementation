"""Defines a neural transducer."""
from typing import Any, Dict, List, Optional, Tuple, Union
import dataclasses
import functools
import heapq
import argparse

import torch
import numpy as np

from trans import optimal_expert
from trans import vocabulary
from trans.actions import ConditionalCopy, ConditionalDel, ConditionalIns, \
    ConditionalSub, Edit, EndOfSequence, GenerativeEdit, BeginOfSequence
from trans.vocabulary import BEGIN_WORD, COPY, DELETE, END_WORD, PAD, \
    FeatureVocabularies
from trans import ENCODER_MAPPING


MAX_ACTION_SEQ_LEN = 150
MAX_INPUT_SEQ_LEN = 100


@functools.total_ordering
@dataclasses.dataclass
class Output:
    action_history: List[Any]
    output: Union[str, List[str]]
    log_p: float
    losses: Optional[torch.tensor] = None

    def __lt__(self, other):
        return self.log_p < other.log_p

    def __eq__(self, other):
        return self.log_p == other.log_p


@dataclasses.dataclass
class Hypothesis:
    action_history: torch.tensor
    alignment: torch.tensor
    decoder: Tuple[torch.tensor, torch.tensor]
    negative_log_p: torch.tensor
    output: List[str]


@functools.total_ordering
@dataclasses.dataclass
class Expansion:
    action: Any
    decoder: Tuple[torch.tensor, torch.tensor]
    from_hypothesis: Hypothesis
    negative_log_p: float

    def __lt__(self, other):
        return self.negative_log_p < other.negative_log_p

    def __eq__(self, other):
        return self.negative_log_p == other.negative_log_p


class Transducer(torch.nn.Module):
    def __init__(self, vocab: vocabulary.Vocabularies,
                 expert: optimal_expert.Expert, args: argparse.Namespace):

        super().__init__()
        self.device = torch.device(args.device)

        self.vocab = vocab
        self.optimal_expert = expert

        self.number_characters = len(vocab.characters)
        self.number_actions = len(vocab.actions)
        self.substitutions = self.vocab.substitutions
        self.inserts = self.vocab.insertions

        self.dec_layers = args.dec_layers
        self.dec_hidden_dim = args.dec_hidden_dim

        # encoder
        self.char_lookup = torch.nn.Embedding(
            num_embeddings=self.number_characters,
            embedding_dim=args.char_dim,
            device=self.device,
            padding_idx=PAD
        )
        if args.enc_type == 'transformer':
            torch.nn.init.normal_(self.char_lookup.weight, mean=0, std=args.char_dim ** -0.5)
            torch.nn.init.constant_(self.char_lookup.weight[PAD], 0)

        self.enc = ENCODER_MAPPING[args.enc_type](args)

        decoder_input_dim = self.enc.output_size + args.action_dim

        # feature encoder if required
        if isinstance(vocab, FeatureVocabularies):
            self.has_features = True
            self.number_features = len(vocab.features)
            self.feat_lookup = torch.nn.Embedding(
                num_embeddings=self.number_features,
                embedding_dim=args.feat_dim,
                device=self.device,
                padding_idx=PAD,
            )
            decoder_input_dim += args.feat_dim
        else:
            self.has_features = False
            self.number_features = None
            self.feat_lookup = None

        # decoder
        self.act_lookup = torch.nn.Embedding(
            num_embeddings=self.number_actions,
            embedding_dim=args.action_dim,
            device=self.device,
            padding_idx=PAD
        )

        self.dec = torch.nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=args.dec_hidden_dim,
            num_layers=args.dec_layers,
            device=self.device,
        )

        self._h0_c0 = None

        # classifier
        self.W = torch.nn.Linear(
            in_features=args.dec_hidden_dim,
            out_features=self.number_actions,
            device=self.device,

        )

        # maps action index to alignment update
        alignment_update = [0] * self.number_actions
        for i, action in enumerate(self.vocab.actions.i2w):
            if isinstance(action,
                          (ConditionalCopy, ConditionalDel, ConditionalSub)):
                alignment_update[i] = 1
        self.alignment_update = torch.tensor(alignment_update, device=self.device)

        # lookup for valid actions (given length of encoder suffix)
        self.valid_actions_lookup = torch.stack(
            [self.compute_valid_actions(i)
             for i in range(MAX_INPUT_SEQ_LEN)],
            dim=0).unsqueeze(dim=0)

    @property
    def h0_c0(self):
        return self._h0_c0

    @h0_c0.setter
    def h0_c0(self, batch_size):
        if not self._h0_c0 or \
                (batch_size and self._h0_c0[0].size(1) != batch_size):
            self._h0_c0 = (
                torch.zeros((self.dec_layers, batch_size, self.dec_hidden_dim), device=self.device),
                torch.zeros((self.dec_layers, batch_size, self.dec_hidden_dim), device=self.device),
            )

    def input_embedding(self, input_: torch.tensor, is_training: bool) -> torch.tensor:
        """Returns a list of character embeddings for the input.

        Args:
            input_: The encoded input string(s).
            is_training: is_training: Bool indicating whether model is in training or not. If True, UNK words are represented as average trained embeddings.

        Returns:
            The corresponding embeddings.
            """
        if input_.dim() == 1:
            input_tensor = input_.unsqueeze(dim=0)
        emb = self.char_lookup(input_)

        if not is_training:

            unk_indices = input_ >= self.number_characters
            if unk_indices.sum().item() > 0:
                # UNK is the average of trained embeddings (excluding UNK)
                ids_tensor = torch.tensor(
                    range(1, self.number_characters),
                    dtype=torch.int,
                    device=self.device,
                )
                unk = self.char_lookup(ids_tensor).mean(dim=0)
                emb[unk_indices] = unk

        return torch.transpose(emb, 0, 1)

    def feature_embedding(self, features: Optional[torch.Tensor], is_training: bool = False) -> Optional[torch.Tensor]:
        """Computes an embedding of the all input features."""
        if not self.has_features:
            return None

        emb = self.feat_lookup(features)  # (batch_size x features x feat_dim)

        if not is_training:

            unk_indices = features >= self.number_features
            if unk_indices.sum().item() > 0:
                # UNK is the average of trained embeddings (excluding UNK)
                ids_tensor = torch.tensor(
                    range(4, self.number_features),
                    dtype=torch.int,
                    device=self.device,
                )
                unk = self.feat_lookup(ids_tensor).mean(dim=0)
                emb[unk_indices] = unk

        return emb.sum(dim=1).unsqueeze(dim=0)  # (1 x batch_size x feat_dim)

    def compute_valid_actions(self, length_encoder_suffix: int) -> torch.tensor:
        """Computes the valid actions for a given encoder suffix as a boolean mask.

        Args:
            length_encoder_suffix: The length of the encoder suffix.

        Returns:
            The boolean mask for the given length."""
        valid_actions = torch.full((self.number_actions,), False,
                                   dtype=torch.bool, device=self.device)
        valid_actions[END_WORD] = True
        valid_actions[self.inserts] = True
        if length_encoder_suffix > 1:
            valid_actions[[COPY, DELETE]] = True
            valid_actions[self.substitutions] = True
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

    def mark_as_invalid(self, logits: torch.tensor,
                        valid_actions_mask: torch.tensor) -> torch.tensor:
        """Mark all logits for all non-valid actions as such (i.e., they are set to -infinity).

        Args:
            logits: The logits.
            valid_actions_mask: A boolean mask indicating all valid actions for all tokens in the batch.
        Returns:
            The 'corrected' logits."""
        log_validity = torch.full(
            logits.size(),
            -np.inf,
            device=self.device,
        )  # All actions invalid by default.
        log_validity[valid_actions_mask] = 0.
        return logits + log_validity

    def log_softmax(self, logits: torch.tensor,
                    valid_actions_mask: torch.tensor) -> torch.tensor:
        """Applies the log_softmax function to all valid actions.
        Args:
            logits: The logits.
            valid_actions_mask: A boolean mask indicating all valid actions for all tokens in the batch.

        Returns:
            The logits after marking non-valid actions as such and applying the log_softmax function."""
        logits_valid = self.mark_as_invalid(logits, valid_actions_mask)
        return torch.nn.functional.log_softmax(logits_valid, dim=2)

    def log_sum_softmax_loss(self, logits: torch.tensor,
                             optimal_actions_mask: torch.tensor,
                             valid_actions_mask: torch.tensor) -> torch.tensor:
        """Compute log loss similar to Riezler et al 2000.

        Args:
            logits: The logits for which the loss is computed.
            optimal_actions_mask: A boolean mask indicating the optimal action for all tokens in the batch.
            valid_actions_mask: A boolean mask indicating all valid actions for all tokens in the batch.

        Returns:
            The computed loss."""
        logits_valid = self.mark_as_invalid(logits, valid_actions_mask)
        # padding can be inferred from optimal actions
        # --> if mask only consists of False values
        paddings = ~torch.any(optimal_actions_mask, dim=2)
        logits_valid[paddings] = -np.inf
        logits_optimal = logits_valid.clone()
        logits_optimal[~(valid_actions_mask * optimal_actions_mask)] = -np.inf

        log_sum_selected_terms = torch.logsumexp(
            logits_optimal,
            dim=2,
        )

        normalization_term = torch.logsumexp(
            logits_valid,
            dim=2,
        )

        if paddings.sum() > 0:
            log_sum_selected_terms =\
                torch.where(~paddings, log_sum_selected_terms, torch.tensor(0., device=self.device))
            normalization_term =\
                torch.where(~paddings, normalization_term, torch.tensor(0., device=self.device))

        return log_sum_selected_terms - normalization_term

    def encoder_step(self, encoded_input: torch.tensor, is_training: bool = False) -> torch.tensor:
        """Runs the encoder.

        Args:
            encoded_input: Encoded input character codes.
            is_training: Bool indicating whether model is in training or not.

        Returns:
            Encoder output."""
        input_emb = self.input_embedding(encoded_input, is_training)

        # encoder input: L x B x E
        if isinstance(self.enc, ENCODER_MAPPING['transformer']):
            bidirectional_emb = self.enc(input_emb,
                                         src_key_padding_mask=(encoded_input == PAD))
        else:
            bidirectional_emb, _ = self.enc(input_emb)

        return bidirectional_emb[1:]  # drop BEGIN_WORD

    def decoder_step(self, encoder_output: torch.tensor,
                     feature_embedding: Optional[torch.tensor],
                     decoder_cell_state: torch.tensor,
                     alignment: torch.tensor,
                     action_history: torch.tensor) -> torch.tensor:
        """Runs the decoder.

        Args:
            encoder_output: The encoder output.
            feature_embedding: Optional feature embedding, the same for all decoder steps.
            decoder_cell_state: The initial decoder cell state.
            alignment: The alignment for all sequences in the batch. This tensor is of shape (L x B) x 1.
            action_history: The action history.

        Returns:
            Decoder output."""
        # build decoder input
        batch_size = encoder_output.size(1)
        input_char_embedding = encoder_output \
            [alignment, torch.tensor([i for i in range(batch_size) for _ in range(len(alignment) // batch_size)],
                                     device=self.device)].unsqueeze(dim=0)
        input_char_embedding = torch.reshape(input_char_embedding,
                                             (batch_size, len(alignment) // batch_size, -1)).transpose(0, 1)
        previous_action_embedding = self.act_lookup(action_history)

        decoder_inputs = [input_char_embedding, previous_action_embedding]
        if self.has_features:
            # Repeats the feature embedding along the decoder steps dimension.
            number_of_decoder_steps = previous_action_embedding.shape[0]
            broadcast_feature_embedding = feature_embedding.\
                repeat((number_of_decoder_steps, 1, 1))
            decoder_inputs.append(broadcast_feature_embedding)

        decoder_input = torch.cat(decoder_inputs, dim=2)

        return self.dec(decoder_input, decoder_cell_state)

    def calculate_actions(self, decoder_output: torch.tensor, valid_actions_mask: torch.tensor)\
            -> Tuple[torch.tensor, torch.tensor]:
        """Calculates the optimal actions (by choosing the max arguments) and log probabilites given the decoder
        output and valid actions for this step.

        Args:
            decoder_output: The output of the decoder.
            valid_actions_mask: A boolean mask indicating all valid actions for all tokens in the batch.

        Returns:
            tuple: A tuple containing:

                actions: The actions.
                log_probabilites: The log probabilites of all actions.
        """
        logits = self.W(decoder_output)
        log_probs = self.log_softmax(logits, valid_actions_mask)
        actions = torch.argmax(log_probs, dim=2)

        return actions, log_probs

    def training_step(self, encoded_input: torch.tensor,
                      encoded_features: Optional[torch.tensor],
                      action_history: torch.tensor,
                      alignment_history: torch.tensor,
                      optimal_actions_mask: torch.tensor,
                      valid_actions_mask: torch.tensor,
                      ) -> torch.tensor:
        """Run a training step and return the respective loss for all sequences in the batch.

        Args:
            encoded_input: Encoded input character codes.
            encoded_features: Optional encoded features.
            action_history: The action history for all sequences. During training this is based on the optimal actions (from the expert).
            alignment_history: The alignment history for all sequences. During training this is based on the optimal alignment (from the expert).
            optimal_actions_mask: A boolean mask indicating the optimal action for all tokens in the batch.
            valid_actions_mask: A boolean mask indicating all valid actions for all tokens in the batch.

        Returns:
            The loss for sequences in the batch. The loss is calculated on sequence-level, i.e., for each sequence
            a single gradient is produced."""
        batch_size = encoded_input.size()[0]

        # adjust initial decoder states if batch_size has changed
        self.h0_c0 = batch_size

        # run encoder
        bidirectional_emb = self.encoder_step(encoded_input, True)

        # compute feature embedding
        feature_emb = self.feature_embedding(encoded_features, True)

        # run decoder & classifier
        decoder_output, _ = self.decoder_step(
            bidirectional_emb, feature_emb, self.h0_c0,
            alignment_history, action_history)
        logits = self.W(decoder_output)

        # compute losses
        # the loss for each seq in the batch is divided by the nr of non-padding elements
        # --> loss per seq = avg. loss per token in seq
        true_action_lengths = action_history.size(0) - (action_history == PAD).sum(dim=0)
        losses = self.log_sum_softmax_loss(logits, optimal_actions_mask, valid_actions_mask)
        losses = -losses.sum(dim=0) / true_action_lengths

        return losses

    def transduce(self, input_: List[List[str]], encoded_input: torch.tensor,
                  encoded_features: Optional[torch.tensor]) -> Output:
        """Runs the transducer for greedy decoding.

        Args:
            input_: Input string.
            encoded_input: Tensor with integer character codes with dimensions (B x L x E).
            encoded_features: Optional tensor integer feature codes (padded and batched).

        Returns:
            An Output object holding the decoded input."""
        batch_size = encoded_input.size()[0]

        # adjust initial decoder states if batch_size has changed
        self.h0_c0 = batch_size

        # initialize state variables
        alignment = torch.full((batch_size,), 0, device=self.device)
        action_history = torch.tensor([[[BEGIN_WORD]] * batch_size],
                                      device=self.device, dtype=torch.int)
        log_p = torch.full((1, batch_size), 0.0, device=self.device)
        true_input_lengths = torch.tensor(
            # +1 because end word is not included in input
            [len(i) + 1 for i in input_], device=self.device)

        # run encoder
        bidirectional_emb = self.encoder_step(encoded_input)

        # compute feature embedding
        feature_emb = self.feature_embedding(encoded_features)

        # initial cell state for decoder
        decoder = self.h0_c0

        # decoding is continued until all sequences
        # in the batch have "found" an end word
        def continue_decoding():
            return torch.any(action_history == END_WORD, dim=2).sum() < batch_size

        while continue_decoding() and action_history.size(2) <= MAX_ACTION_SEQ_LEN:
            valid_actions_mask = self.valid_actions_lookup[:, true_input_lengths - alignment]

            # run decoder
            decoder_output, decoder = self.decoder_step(
                bidirectional_emb, feature_emb, decoder,
                alignment, action_history[:, :, -1])

            # get actions
            actions, log_probs = self.calculate_actions(decoder_output, valid_actions_mask)

            # update states
            log_p += log_probs[:, torch.arange(batch_size), actions.squeeze(dim=0)]
            action_history = torch.cat(
                (action_history, actions.unsqueeze(dim=2)),
                dim=2
            )
            alignment = alignment + self.alignment_update[actions.squeeze(dim=0)]

        # adjust log_p
        # --> return the token avg. of all seqs in the batch
        true_action_lengths = action_history.size(2) - (action_history == PAD).sum(dim=2)
        log_p = torch.mean(log_p.sum(dim=0) / true_action_lengths).item()

        # trim action history
        # --> first element is not considered (begin-of-sequence-token)
        # --> and only token up to the first end-of-sequence-token (including it)
        action_history = [seq[1:(seq.index(EndOfSequence()) + 1 if EndOfSequence() in seq else -1)]
                          for seq in action_history.squeeze(dim=0).tolist()]

        return Output(action_history, self.decode_encoded_output(input_, action_history),
                      log_p, None)

    def decode_encoded_output(self, input_: List[List[str]], encoded_output: List[List[int]]) -> List[str]:
        """Decode a list of encoded output sequences given their string input.

        Args:
            input_: Input string.
            encoded_output: Holds the encoded integers output (--> encoded actions) corresponding to the input sequences.

        Returns:
            A list of the decoded strings."""
        output = []
        for i, seq in enumerate(encoded_output):
            decoded_seq = ""
            alignment = 0
            for a in seq:
                char_, alignment, _ = self.decode_single_action(input_[i], a, alignment)
                decoded_seq += char_
            output.append("".join(decoded_seq))

        return output

    def decode_single_action(self, input_: Union[str, List[str]], action: Union[int, Edit],
                             alignment: Union[int, torch.tensor]) -> Tuple[str, int, bool]:
        """Decodes a single char, given the corresponding input string, action and alignment.

        Args:
            input_: The input string.
            action: The action, may be encoded or not.
            alignment: Position of control in the input string.

        Returns:
            tuple: A tuple containing:
                char_: The decoded char.
                alignment: The updated alignment.
                stop: A bool indicating whether the end of sequence is reached.
            """
        if isinstance(action, int):
            action = self.vocab.decode_action(action)
        stop = False

        if isinstance(action, ConditionalCopy):
            char_ = input_[alignment]
            alignment += 1
        elif isinstance(action, ConditionalDel):
            char_ = ""
            alignment += 1
        elif isinstance(action, ConditionalIns):
            char_ = action.new
        elif isinstance(action, ConditionalSub):
            char_ = action.new
            alignment += 1
        elif isinstance(action, EndOfSequence):
            char_ = ""
            stop = True
        elif isinstance(action, BeginOfSequence):
            char_ = ""
        else:
            raise ValueError(f"Unknown action: {action}.")

        return char_, alignment, stop

    def beam_search_decode(self, input_: str, encoded_input: torch.tensor,
                           encoded_features: Optional[torch.tensor],
                           beam_width: int) -> List[Output]:
        """Runs the transducer with beam search.

        Args:
            input_: Input string.
            encoded_input: List of integer character codes.
            encoded_features: Optional tensor of feature codes.
            beam_width: Width of the beam search.

        Returns:
            A list holding the output of the best search paths.
        """
        # adjust initial decoder states if batch_size has changed
        self.h0_c0 = 1  # nothing else possible at the moment

        # run encoder
        bidirectional_emb = self.encoder_step(encoded_input)

        # compute feature embedding
        feature_emb = self.feature_embedding(encoded_features)

        input_length = len(input_) + 1  # +1 because of begin-of-seq-token

        beam: List[Hypothesis] = [
            Hypothesis(action_history=torch.tensor([[BEGIN_WORD]], device=self.device),
                       alignment=torch.tensor([0], device=self.device),
                       decoder=self.h0_c0,
                       negative_log_p=torch.tensor(0., device=self.device),
                       output=[])]

        hypothesis_length = 0
        complete_hypotheses = []

        while beam and beam_width > 0 and hypothesis_length <= MAX_ACTION_SEQ_LEN:

            expansions: List[Expansion] = []

            for hypothesis in beam:

                length_encoder_suffix = max(input_length - hypothesis.alignment, torch.tensor([0], device=self.device))
                valid_actions_mask = self.valid_actions_lookup[:, length_encoder_suffix]
                # decoder
                decoder_output, decoder = self.decoder_step(bidirectional_emb,
                                                            feature_emb,
                                                            hypothesis.decoder,
                                                            hypothesis.alignment,
                                                            hypothesis.action_history[-1].unsqueeze(dim=0))
                logits = self.W(decoder_output)
                log_probs = self.log_softmax(logits, valid_actions_mask)

                for action in torch.arange(0, valid_actions_mask.size(2), device=self.device):
                    if not valid_actions_mask[0, 0, action]:
                        continue
                    log_p = hypothesis.negative_log_p - \
                            log_probs[0, 0, action]  # min heap, so minus

                    heapq.heappush(expansions,
                                   Expansion(action.reshape(1, -1), decoder,
                                             hypothesis, log_p))

            beam: List[Hypothesis] = []

            for _ in range(beam_width):

                expansion: Expansion = heapq.heappop(expansions)
                from_hypothesis = expansion.from_hypothesis
                action = expansion.action
                action_history = from_hypothesis.action_history
                action_history = torch.cat(
                    (action_history, action)
                )
                output = list(from_hypothesis.output)

                # execute the action to update the transducer state
                action = self.vocab.decode_action(action.item())

                if isinstance(action, EndOfSequence):
                    # 1. COMPLETE HYPOTHESIS, REDUCE BEAM
                    complete_hypothesis = Output(
                        action_history=action_history.squeeze(dim=1).tolist()[1:],
                        output="".join(output),
                        log_p=-expansion.negative_log_p.item())  # undo min heap minus

                    complete_hypotheses.append(complete_hypothesis)
                    beam_width -= 1
                else:
                    # 2. EXECUTE ACTION AND ADD FULL HYPOTHESIS TO NEW BEAM
                    alignment = from_hypothesis.alignment.clone()

                    char_, alignment, _ = self.decode_single_action(input_, action, alignment)
                    if char_ != "":
                        output.append(char_)

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
                    action_history=hypothesis.action_history.squeeze(dim=1).tolist()[1:],
                    output="".join(hypothesis.output),
                    log_p=-hypothesis.negative_log_p.item())  # undo min heap minus

                complete_hypotheses.append(complete_hypothesis)

        complete_hypotheses.sort(reverse=True)
        return complete_hypotheses
