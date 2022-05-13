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
        for sample in batch:
            output = decoding(sample)
            prediction = output.output
            predictions.append(f"{sample.input}\t{prediction}")
            if prediction == sample.target:
                correct += 1
            loss += output.log_p / (len(output.action_history) - 1)
            if j > 0 and j % 500 == 0:
                logging.info("\t\t...%d samples", j)
            j += 1
    logging.info("\t\t...%d samples", j)

    return utils.DecodingOutput(accuracy=correct / len(data_loader.dataset),
                                loss=-loss / len(data_loader.dataset),
                                predictions=predictions)


def inverse_sigmoid_schedule(k: int):
    """Probability of sampling an action from the model as function of epoch."""
    return lambda epoch: (1 - k / (k + np.exp(epoch / k)))


def main(args: argparse.Namespace):

    dargs = args.__dict__
    for key, value in dargs.items():
        logging.info("%s: %s", str(key).ljust(15), value)

    os.makedirs(args.output)

    if args.pytorch_seed is not None:
        torch.manual_seed(args.pytorch_seed)

        train_generator = torch.Generator()
        train_generator.manual_seed(args.pytorch_seed)

        def train_worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
    else:
        train_generator, train_worker_init_fn = None, None

    if args.nfd:
        logging.info("Will perform training on NFD-normalized data.")
    else:
        logging.info("Will perform training on unnormalized data.")

    vocabulary_ = vocabulary.Vocabularies()

    training_data = utils.Dataset()
    with utils.OpenNormalize(args.train, args.nfd) as f:
        for line in f:
            input_, target = line.rstrip().split("\t", 1)
            encoded_input = vocabulary_.encode_input(input_)
            vocabulary_.encode_actions(target)
            sample = utils.Sample(input_, target, encoded_input)
            training_data.add_samples(sample)
    training_data_loader = training_data.get_data_loader(batch_size=args.batch_size, shuffle=True,
                                                         generator=train_generator,
                                                         worker_init_fn=train_worker_init_fn)

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
            encoded_input = vocabulary_.encode_unseen_input(input_)
            sample = utils.Sample(input_, target, encoded_input)
            development_data.add_samples(sample)
    development_data_loader = development_data.get_data_loader()

    if args.test is not None:
        test_data = utils.Dataset()
        with utils.OpenNormalize(args.test, args.nfd) as f:
            for line in f:
                input_, *optional_target = line.rstrip().split("\t", 1)
                target = optional_target[0] if optional_target else None
                encoded_input = vocabulary_.encode_unseen_input(input_)
                sample = utils.Sample(input_, target, encoded_input)
                test_data.add_samples(sample)
        test_data_loader = test_data.get_data_loader()

    sed_parameters_path = os.path.join(args.output, "sed.pkl")
    sed_aligner = sed.StochasticEditDistance.fit_from_data(
        training_data.samples, em_iterations=args.sed_em_iterations,
        output_path=sed_parameters_path)
    expert = optimal_expert_substitutions.OptimalSubstitutionExpert(sed_aligner)

    transducer_ = transducer.Transducer(vocabulary_, expert, **dargs)  # removed model

    widgets = [progressbar.Bar(">"), " ", progressbar.ETA()]
    train_progress_bar = progressbar.ProgressBar(
        widgets=widgets, maxval=args.epochs).start()

    train_log_path = os.path.join(args.output, "train.log")
    best_model_path = os.path.join(args.output, "best.model")

    with open(train_log_path, "w") as w:
        w.write("epoch\tavg_loss\ttrain_accuracy\tdev_accuracy\n")

    optimizer = torch.optim.Adadelta(transducer_.parameters())
    train_subset_loader = utils.Dataset(training_data.samples[:100]).get_data_loader()
    rollin_schedule = inverse_sigmoid_schedule(args.k)
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
        with utils.Timer():
            train_loss = 0.
            rollin = rollin_schedule(epoch)
            j = 0
            for j, batch in enumerate(training_data_loader):
                losses = []
                transducer_.zero_grad()
                for sample in batch:
                    output = transducer_.transduce(
                        input_=sample.input,
                        encoded_input=sample.encoded_input,
                        target=sample.target,
                        rollin=rollin,
                    )
                    # split losses --> optimize for all optimal actions
                    losses.extend([s for loss in output.losses for s in torch.split(loss, 1)])
                batch_loss = -torch.mean(torch.stack(losses))
                train_loss += batch_loss.item()
                batch_loss.backward()
                optimizer.step()
                if j > 0 and j % 100 == 0:
                    logging.info("\t\t...%d batches", j)
            logging.info("\t\t...%d batches", j + 1)

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

    parser.add_argument("--pytorch-seed", type=int,
                        help="Random seed used by PyTorch.")
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
    parser.add_argument("--k", type=int, default=1,
                        help="k for inverse sigmoid rollin schedule.")
    parser.add_argument("--patience", type=int, default=12,
                        help="Maximal patience for early stopping.")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Maximal number of training epochs.")
    parser.add_argument("--batch-size", type=str, default=5,
                        help="Batch size.")
    parser.add_argument("--sed-em-iterations", type=int, default=10,
                        help="SED EM iterations.")
    parser.add_argument("--device", type=str, default='cpu',
                        help="Device to run training on.")

    args = parser.parse_args()
    main(args)
