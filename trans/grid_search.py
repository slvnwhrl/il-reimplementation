"""CLI for performing grid search."""

import argparse
import json
import subprocess
import shutil
import os
import itertools
import time
import atexit
import re
from typing import Any, Optional, List


BASH_EXECUTABLE = shutil.which("bash")


def cleanup():
    if 'process_list' in globals():
        for p in process_list:
            if p.poll() is None:
                p.kill()


atexit.register(cleanup)


def last_value_from_file(file_path: str, t=float):
    with open(file_path) as f:
        lines = f.readlines()
        return t(lines[-1].split()[-1])


def get_list(var: Any):
    return var if isinstance(var, list) else [var]


def file_name_from_pattern(pattern: str, lang: str, split: str):
    file_name = pattern.replace("LANG", lang)
    file_name = file_name.replace("SPLIT", split)
    return file_name


def run_ensemble(gold: str, systems: List[str], output: str):
    subprocess.Popen([
        "trans-ensemble"
        "--gold", gold,
        "--systems", *systems,
        "--output", output
    ]).wait()


def write_to_results_file(results_file: str, results: List[dict], beam_width: Optional[str] = None):
    with open(results_file, "w") as f:
        for r in sorted(results, key=lambda x: x['dev_greedy'], reverse=True):
            f.write(r['c_dir'] + "\n")
            f.write(f"dev\ngreedy: {r['dev_greedy']}\n")
            if r['dev_beam']:
                f.write(f"{beam_width}: {r['dev_beam']}\n")
            if r['test_greedy']:
                f.write(f"test\ngreedy: {r['test_greedy']}\n")
            if r['test_beam']:
                f.write(f"{beam_width}: {r['test_beam']}\n\n")
            else:
                f.write("\n")


def main(args: argparse.Namespace):
    os.makedirs(args.output, exist_ok=True)

    config_file = open(args.config)
    config_dict = json.load(config_file)

    process_list = []
    for name, grid_config in config_dict["grids"].items():
        os.makedirs(f"{args.output}/{name}")

        nm_pairs = [[(k, v) for v in get_list(grid_config[k])] for k in grid_config]
        combinations = itertools.product(*nm_pairs)

        # parse args
        args_list, comb_dict = [], {}
        for i, c in enumerate(combinations, 1):
            parsed_args, args_dict = [], {}
            for j in c:
                par_name, par_value = j
                if isinstance(par_value, bool) and par_value:
                    parsed_args.append(f"--{par_name}")
                elif isinstance(par_value, (list, tuple)):
                    parsed_args.extend([f"--{par_name}", *[str(v) for v in par_value]])
                elif par_name in ['sed-params', 'precomputed-train', 'vocabulary']:
                    continue
                else:
                    parsed_args.extend([f"--{par_name}", str(par_value)])
                args_dict[par_name] = par_value
            args_list.append(parsed_args)
            comb_dict[i] = args_dict

        with open(f"{args.output}/{name}/combinations.json", "w") as f:
            json.dump(comb_dict, f, indent=4)

        # train
        for i, args_ in enumerate(args_list, 1):
            for lang in config_dict['data']['languages']:
                for j in range(1, config_dict['runs_per_model']+1):
                    # reset ext_args
                    ext_args = args_.copy()

                    output = f"{args.output}/{name}/{lang}/{i}/{i}.{j}"

                    for par in ['sed-params', 'vocabulary']:
                        if par in grid_config and lang in grid_config[par]:
                            ext_args.extend(
                                [
                                    "--"+par, grid_config[par][lang]
                                ]
                            )

                    # create file names from pattern
                    dev_file = file_name_from_pattern(config_dict['data']['pattern'], lang, 'dev')
                    test_file = file_name_from_pattern(config_dict['data']['pattern'], lang, 'test')

                    dev = f"{config_dict['data']['path']}/{dev_file}"
                    test = f"{config_dict['data']['path']}/{test_file}"

                    # for train it's only needed if --train-precomputed is not specified
                    if not ('precomputed-train' in grid_config and lang in grid_config['precomputed-train']):
                        train_file = file_name_from_pattern(config_dict['data']['pattern'], lang, 'train')
                        train = f"{config_dict['data']['path']}/{train_file}"
                        train_par = ("--train", train)
                    else:
                        train_par = ("--precomputed-train", grid_config['precomputed-train'][lang])

                    ext_args.extend(
                        [
                            "--output", output,
                            *train_par,
                            "--dev", dev
                         ]
                    )

                    if os.path.exists(test):
                        ext_args.extend(
                            [
                                "--test", test
                            ]
                        )

                    p = subprocess.Popen(["trans-train"]+ext_args, bufsize=0)
                    process_list.append(p)

                    if len(process_list) < args.parallel_jobs:
                        continue

                    while len(process_list) >= args.parallel_jobs:
                        process_list = [p for p in process_list if p.poll() is
                                        None]
                        # check every few seconds
                        time.sleep(5)

    # all trainings in progress, stay in script so all processes can be aborted
    while len(process_list) > 0:
        process_list = [p for p in process_list if p.poll() is
                        None]
        # check every few seconds
        time.sleep(5)

    # evaluate: average of results per combination and ensemble
    for name, grid_config in config_dict["grids"].items():
        for lang in config_dict["data"]["languages"]:

            results = []  # average of single models
            ensemble_results = []  # ensemble results
            output_path = f"{args.output}/{name}/{lang}"
            dev_file =\
                f"{config_dict['data']['path']}/{file_name_from_pattern(config_dict['data']['pattern'], lang, 'dev')}"
            test_file =\
                f"{config_dict['data']['path']}/{file_name_from_pattern(config_dict['data']['pattern'], lang, 'test')}"

            # level: combination
            for c_dir in os.listdir(output_path):  # c_dir == name of combination (number)
                dev_beam_avg, dev_greedy_avg = 0, 0
                test_beam_avg, test_greedy_avg = 0, 0

                # get beam size
                c_first_run = os.listdir(f"{output_path}/{c_dir}")[0]
                beam_match = re.search(r"beam[0-9]+", " ".join(os.listdir(f"{output_path}/{c_dir}/{c_first_run}")))
                beam_width = beam_match[0] if beam_match else None

                # check if test files are evaluated
                test_match = "test_greedy.eval" in " ".join(os.listdir(f"{output_path}/{c_dir}/{c_first_run}"))

                # level: run per combination
                c_dir_path = f"{output_path}/{c_dir}"  # directory of combination
                n_runs = len(os.listdir(c_dir_path))  # number of runs for combination
                for c_run in os.listdir(c_dir_path):
                    # dev greedy
                    dev_greedy_avg += last_value_from_file(f"{c_dir_path}/{c_run}/dev_greedy.eval")/n_runs

                    # dev beam
                    if beam_width:
                        dev_beam_avg += last_value_from_file(f"{c_dir_path}/{c_run}/dev_{beam_width}.eval")/n_runs

                    if test_match:
                        # test greedy
                        test_greedy_avg += last_value_from_file(f"{c_dir_path}/{c_run}/test_greedy.eval")/n_runs
                        # test beam
                        if beam_width:
                            test_beam_avg += last_value_from_file(f"{c_dir_path}/{c_run}/test_{beam_width}.eval")/n_runs

                result = {
                    'c_dir': c_dir,
                    'dev_greedy': round(dev_greedy_avg, 4),
                    'dev_beam': round(dev_beam_avg, 4) if beam_width else None,
                    'test_greedy': round(test_greedy_avg, 4) if test_match else None,
                    'test_beam': round(test_beam_avg, 4) if test_match and beam_width else None
                }
                results.append(result)

                if args.ensemble:
                    result = {
                        'c_dir': c_dir,
                        'dev_beam': None,
                        'test_greedy': None,
                        'test_beam': None
                    }
                    golds = [('dev', dev_file), ('test', test_file)] if test_match else [('dev', dev_file)]
                    for split, gold_file in golds:
                        systems =\
                            [f"{c_dir_path}/{c_dir}.{i}/{split}_greedy.predictions" for i in range(1, n_runs+1)]
                        # greedy
                        run_ensemble(gold_file, systems, f"{c_dir_path}/greedy_ensemble")
                        result[f"{split}_greedy"] =\
                            round(last_value_from_file(f"{c_dir_path}/greedy_ensemble/{split}_{n_runs}ensemble.eval"), 4)
                        # beam
                        if beam_width:
                            systems = \
                                [f"{c_dir_path}/{c_dir}.{i}/{split}_{beam_width}.predictions" for i in range(1, n_runs + 1)]
                            run_ensemble(gold_file, systems, f"{c_dir_path}/{beam_width}_ensemble")
                            result[f"{split}_beam"] = \
                                round(last_value_from_file(f"{c_dir_path}/{beam_width}_ensemble/{split}_{n_runs}ensemble.eval"), 4)
                    ensemble_results.append(result)

            # write to results text file
            write_to_results_file(f"{args.output}/{name}/{lang}/results.txt", results, beam_width)

            if args.ensemble:
                write_to_results_file(f"{args.output}/{name}/{lang}/ensemble_results.txt", ensemble_results, beam_width)


def cli_main():
    parser = argparse.ArgumentParser(
        description="Grid search.")

    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output directory.")
    parser.add_argument("--parallel-jobs", type=int,
                        default=30, help="Max number of parallel trainings.")
    parser.add_argument("--ensemble", action="store_true",
                        help="Produce ensemble results.")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
