"""Trains a grapheme-to-phoneme neural transducer."""
from typing import List, Optional, Union

import argparse
import logging
import os
import random
import sys

import progressbar

import torch
import numpy as np

from trans import optimal_expert_substitutions
from trans import sed
from trans import transducer
from trans import utils
from trans import vocabulary

random.seed(1)


def decode(transducer_: transducer.Transducer, data_loader: torch.utils.data.DataLoader,
           beam_width: int = 1) -> utils.DecodingOutput:
    if beam_width == 1:
        decoding = lambda s: \
            transducer_.transduce(s.input, s.encoded_input)
    else:
        decoding = lambda s: \
            transducer_.beam_search_decode(s.input, s.encoded_input,
                                           beam_width)[0]
    predictions = []
    loss = 0
    correct = 0
    j = 0
    for batch in data_loader:
        output = decoding(batch)
        inputs, targets = batch.input, batch.target
        for i, p in enumerate(output.output):
            predictions.append(f"{inputs[i]}\t{p}")
            if p == targets[i]:
                correct += 1
        loss += output.log_p
        if j > 0 and j % 100 == 0:
            logging.info("\t\t...%d batches", j)
        j += 1
    logging.info("\t\t...%d batches", j)

    return utils.DecodingOutput(accuracy=correct / len(data_loader.dataset),
                                loss=-loss / len(data_loader.dataset),
                                predictions=predictions)


def inverse_sigmoid_schedule(k: int):
    """Probability of sampling an action from the model as function of epoch."""
    return lambda epoch: (1 - k / (k + np.exp(epoch / k)))


def precompute_from_expert(s: utils.Sample, transducer_: transducer.Transducer) -> None:
    """ Precompute the optimal policy (optimal and valid actions as well as the alignment) from the expert.

    Args:
        s: A data sample.
        transducer_: The transducer object holding the expert.

    Returns:
        None
    """
    alignment_history = [0]
    action_history = [[vocabulary.BEGIN_WORD]]
    output = []
    a = 0
    stop = False

    # continue until end-of-sequence-token is found
    # (or max seq len is reached)
    while not stop and len(output) <= transducer.MAX_ACTION_SEQ_LEN:
        actions = transducer_.expert_rollout(s.input, s.target, a, output)
        # todo: allow optimization of multiple target actions
        action_history.append([actions[0]])

        char_, a, stop = transducer_.decode_char(s.input, actions[0], a)
        alignment_history.append(a)
        if char_ != "":
            output.append(char_)

    optimal_actions_mask = torch.full(
        (len(action_history)-1, transducer_.number_actions),
        False, dtype=torch.bool, device=args.device)
    seq_pos, emb_pos = zip(*[(s-1, a) for s in range(1, len(action_history))
                             for a in action_history[s]])
    optimal_actions_mask[seq_pos, emb_pos] = True
    s.optimal_actions_mask = optimal_actions_mask

    # now this is a crucial part: the last alignment index as well as the last action
    # are irrelevant and changing these lines will mess up training
    s.alignment_history = torch.tensor(alignment_history[:-1], device=args.device)
    s.action_history = torch.tensor(action_history[:-1], device=args.device).squeeze(dim=1)

    valid_actions_mask = torch.stack(
        # + 1 is needed to compensate for lack of end-of-seq-token
        # :-1 for same reason as above
        [transducer_.compute_valid_actions(len(s.input) + 1 - a) for a in alignment_history[:-1]], dim=0)
    s.valid_actions_mask = valid_actions_mask


def main(args: argparse.Namespace):
    dargs = args.__dict__
    for key, value in dargs.items():
        logging.info("%s: %s", str(key).ljust(15), value)

    os.makedirs(args.output)

    if args.nfd:
        logging.info("Will perform training on NFD-normalized data.")
    else:
        logging.info("Will perform training on unnormalized data.")

    vocabulary_ = vocabulary.Vocabularies()

    training_data = utils.Dataset()
    with utils.OpenNormalize(args.train, args.nfd) as f:
        for line in f:
            input_, target = line.rstrip().split("\t", 1)
            encoded_input = torch.tensor(vocabulary_.encode_input(input_),
                                         device=args.device)
            vocabulary_.encode_actions(target)
            sample = utils.Sample(input_, target, encoded_input)
            training_data.add_samples(sample)

    logging.info("%d actions: %s", len(vocabulary_.actions),
                 vocabulary_.actions)
    logging.info("%d chars: %s", len(vocabulary_.characters),
                 vocabulary_.characters)
    vocabulary_path = os.path.join(args.output, "vocabulary.pkl")
    vocabulary_.persist(vocabulary_path)
    logging.info("Wrote vocabulary to %s.", vocabulary_path)

    development_data = utils.Dataset()
    with utils.OpenNormalize(args.dev, args.nfd) as f:
        for line in f:
            input_, target = line.rstrip().split("\t", 1)
            encoded_input = torch.tensor(vocabulary_.encode_unseen_input(input_),
                                         device=args.device)
            sample = utils.Sample(input_, target, encoded_input)
            development_data.add_samples(sample)
    development_data_loader = development_data.get_data_loader()

    if args.test is not None:
        test_data = utils.Dataset()
        with utils.OpenNormalize(args.test, args.nfd) as f:
            for line in f:
                input_, *optional_target = line.rstrip().split("\t", 1)
                target = optional_target[0] if optional_target else None
                encoded_input = torch.tensor(vocabulary_.encode_unseen_input(input_),
                                             device=args.device)
                sample = utils.Sample(input_, target, encoded_input)
                test_data.add_samples(sample)
        test_data_loader = test_data.get_data_loader()

    sed_parameters_path = os.path.join(args.output, "sed.pkl")
    sed_aligner = sed.StochasticEditDistance.fit_from_data(
        training_data.samples, em_iterations=args.sed_em_iterations,
        output_path=sed_parameters_path)
    expert = optimal_expert_substitutions.OptimalSubstitutionExpert(sed_aligner)

    transducer_ = transducer.Transducer(vocabulary_, expert, **dargs)

    # precompute from expert
    for s in training_data.samples:
        precompute_from_expert(s, transducer_)
    training_data_loader = training_data.get_data_loader(is_training=True, batch_size=args.batch_size)

    widgets = [progressbar.Bar(">"), " ", progressbar.ETA()]
    train_progress_bar = progressbar.ProgressBar(
        widgets=widgets, maxval=args.epochs).start()

    train_log_path = os.path.join(args.output, "train.log")
    best_model_path = os.path.join(args.output, "best.model")

    with open(train_log_path, "w") as w:
        w.write("epoch\tavg_loss\ttrain_accuracy\tdev_accuracy\n")

    optimizer = torch.optim.Adadelta(transducer_.parameters())
    train_subset_loader = utils.Dataset(training_data.samples[:100]).get_data_loader()
    # rollin_schedule = inverse_sigmoid_schedule(args.k)
    max_patience = args.patience

    logging.info("Training for a maximum of %d with a maximum patience of %d.",
                 args.epochs, max_patience)
    logging.info("Number of train batches: %d.", len(training_data_loader))

    best_train_accuracy = 0
    best_dev_accuracy = 0
    best_epoch = 0
    patience = 0

    for epoch in range(args.epochs):

        logging.info("Training...")
        transducer_.train()
        transducer_.zero_grad()
        with utils.Timer():
            train_loss = 0.
            # rollin not implemented at the moment
            # rollin = rollin_schedule(epoch)
            j = 0
            for j, batch in enumerate(training_data_loader):
                losses = transducer_.training_step(encoded_input=batch.encoded_input,
                                                   action_history=batch.action_history,
                                                   alignment_history=batch.alignment_history,
                                                   optimal_actions_mask=batch.optimal_actions_mask,
                                                   valid_actions_mask=batch.valid_actions_mask)
                train_loss += torch.mean(losses.squeeze(dim=0)).item()  # mean per batch
                losses.sum().backward()
                if j % args.grad_accumulation == 0:
                    optimizer.step()
                    transducer_.zero_grad()
                if j > 0 and j % 100 == 0:
                    logging.info("\t\t...%d batches", j)
            logging.info("\t\t...%d batches", j + 1)

        # avg. loss per sample
        avg_loss = train_loss / len(training_data_loader)
        logging.info("Average train loss: %.4f.", avg_loss)

        transducer_.eval()
        with torch.no_grad():
            logging.info("Evaluating on training data subset...")
            with utils.Timer():
                train_accuracy = decode(transducer_, train_subset_loader).accuracy

            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy

            patience += 1

            logging.info("Evaluating on development data...")
            with utils.Timer():
                decoding_output = decode(transducer_, development_data_loader)
                dev_accuracy = decoding_output.accuracy
                avg_dev_loss = decoding_output.loss

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_epoch = epoch
            patience = 0
            logging.info("Found best dev accuracy %.4f.", best_dev_accuracy)
            torch.save(transducer_.state_dict(), best_model_path)
            logging.info("Saved new best model to %s.", best_model_path)

        logging.info(
            f"Epoch {epoch} / {args.epochs - 1}: train loss: {avg_loss:.4f} "
            f"dev loss: {avg_dev_loss:.4f} train acc: {train_accuracy:.4f} "
            f"dev acc: {dev_accuracy:.4f} best train acc: {best_train_accuracy:.4f} "
            f"best dev acc: {best_dev_accuracy:.4f} best epoch: {best_epoch} "
            f"patience: {patience} / {max_patience - 1}"
        )

        log_line = f"{epoch}\t{avg_loss:.4f}\t{train_accuracy:.4f}\t{dev_accuracy:.4f}\n"
        with open(train_log_path, "a") as a:
            a.write(log_line)

        if patience == max_patience:
            logging.info("Out of patience after %d epochs.", epoch + 1)
            train_progress_bar.finish()
            break

        train_progress_bar.update(epoch)

    logging.info("Finished training.")

    if not os.path.exists(best_model_path):
        sys.exit(0)

    transducer_ = transducer.Transducer(vocabulary_, expert, **dargs)
    transducer_.load_state_dict(torch.load(best_model_path))

    transducer_.eval()
    with torch.no_grad():
        evaluations = [(development_data_loader, "dev")]
        if args.test is not None:
            evaluations.append((test_data_loader, "test"))
        for data, dataset_name in evaluations:
            logging.info("Evaluating best model on %s data using beam search "
                         "(beam width %d)...", dataset_name, args.beam_width)
            with utils.Timer():
                beam_decoding = decode(transducer_, data, args.beam_width)
            utils.write_results(beam_decoding.accuracy,
                                beam_decoding.predictions, args.output,
                                args.nfd, dataset_name, args.beam_width,
                                dargs=dargs)
            logging.info("Evaluating best model on %s data using greedy decoding"
                         , dataset_name)
            with utils.Timer():
                greedy_decoding = decode(transducer_, data)
            utils.write_results(greedy_decoding.accuracy,
                                greedy_decoding.predictions, args.output,
                                args.nfd, dataset_name, dargs=dargs)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Train a g2p neural transducer.")

    '''parser.add_argument("--dynet-seed", type=int, required=True,
                        help="DyNET random seed.")'''  # TODO: how to replace if at all?
    parser.add_argument("--train", type=str, required=True,
                        help="Path to train set data.")
    parser.add_argument("--dev", type=str, required=True,
                        help="Path to development set data.")
    parser.add_argument("--test", type=str,
                        help="Path to development set data.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("--nfd", action="store_true",
                        help="Train on NFD-normalized data. Write out in NFC.")
    parser.add_argument("--char-dim", type=int, default=100,
                        help="Character peak_embedding dimension.")
    parser.add_argument("--action-dim", type=int, default=100,
                        help="Action peak_embedding dimension.")
    parser.add_argument("--enc-hidden-dim", type=int, default=200,
                        help="Encoder LSTM state dimension.")
    parser.add_argument("--dec-hidden-dim", type=int, default=200,
                        help="Decoder LSTM state dimension.")
    parser.add_argument("--enc-layers", type=int, default=1,
                        help="Number of encoder LSTM layers.")
    parser.add_argument("--dec-layers", type=int, default=1,
                        help="Number of decoder LSTM layers.")
    parser.add_argument("--beam-width", type=int, default=4,
                        help="Beam width for beam search decoding.")
    # parser.add_argument("--k", type=int, default=1,
    #                     help="k for inverse sigmoid rollin schedule.")
    parser.add_argument("--patience", type=int, default=12,
                        help="Maximal patience for early stopping.")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Maximal number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Batch size.")
    parser.add_argument("--grad-accumulation", type=int, default=1,
                        help="Gradient accumulation.")
    parser.add_argument("--sed-em-iterations", type=int, default=10,
                        help="SED EM iterations.")
    parser.add_argument("--device", type=str, default='cpu',
                        help="Device to run training on.")

    args = parser.parse_args()
    main(args)
