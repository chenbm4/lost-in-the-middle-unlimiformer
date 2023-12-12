#!/usr/bin/env python3
"""Given a data file with LM QA predictions, evaluate the predictions.
"""
import argparse
import json
import logging
import statistics
import sys
import os
from copy import deepcopy

import matplotlib.pyplot as plt
from tqdm import tqdm
from xopen import xopen

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(src_dir)

from lost_in_the_middle.metrics import best_subspan_em

logger = logging.getLogger(__name__)

METRICS = [
    (best_subspan_em, "best_subspan_em"),
]


def main(input_path, output_path, score_by_new_gold_index):
    all_examples = []
    examples_by_new_gold_index = {}  # To store examples by their new_gold_index
    metric_values_by_new_gold_index = {}  # To store metric values by their new_gold_index

    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            all_examples.append(input_example)
            if score_by_new_gold_index:
                new_gold_index = input_example.get("new_gold_index")
                if new_gold_index is not None:
                    examples_by_new_gold_index.setdefault(new_gold_index, []).append(input_example)

    # Compute and log overall metrics
    log_metrics(all_examples, "Overall")

    if score_by_new_gold_index:
        # Sort and log metrics per new_gold_index
        for new_gold_index in sorted(examples_by_new_gold_index.keys()):
            logger.info(f"new_gold_index: {new_gold_index}")
            metrics = log_metrics(examples_by_new_gold_index[new_gold_index], f"new_gold_index {new_gold_index}")
            metric_values_by_new_gold_index[new_gold_index] = metrics

        plot_metrics(metric_values_by_new_gold_index)


    if output_path:
        # Write examples with metrics to output file
        write_output(all_examples, output_path)


def get_metrics_for_example(example):
    gold_answers = example["answers"]
    model_answer = example["model_answer"]

    # NOTE: we take everything up to the first newline, since otherwise models could hack
    # the metric by simply copying te input context (as the gold answer is guaranteed
    # to occur in the input context).
    model_answer = model_answer.split("\n")[0].strip()

    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
    return (example_metrics, example)


def log_metrics(examples, label):
    logger.info(f"Computing metrics for {label}")
    all_example_metrics = [get_metrics_for_example(example) for example in examples]

    metric_averages = {}

    # Average metrics across examples
    for (_, metric_name) in METRICS:
        average_metric_value = statistics.mean(
            example_metrics[metric_name] for (example_metrics, _) in all_example_metrics
        )
        logger.info(f"{metric_name} ({label}): {average_metric_value}")
        metric_averages[metric_name] = average_metric_value

    return metric_averages


def plot_metrics(metric_values_by_new_gold_index):
    # Plotting
    for metric_name in METRICS:
        _, metric_label = metric_name
        values = [metric_values_by_new_gold_index[idx][metric_label] for idx in sorted(metric_values_by_new_gold_index)]

        plt.figure()
        plt.plot(sorted(metric_values_by_new_gold_index.keys()), values, marker='o')
        plt.title(f'Metric: {metric_label}')
        plt.xlabel('new_gold_index')
        plt.ylabel(metric_label)
        plt.grid(True)
        plt.show()


def write_output(examples, output_path):
    with xopen(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with model predictions and answers.", required=True)
    parser.add_argument(
        "--output-path",
        help="Path to write data with model predictions, answers, and scores.",
    )
    parser.add_argument(
        "--score-by-new-gold-index",
        action="store_true",
        help="Calculate scores per new_gold_index."
    )
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.output_path,
        args.score_by_new_gold_index,
    )
    logger.info("finished running %s", sys.argv[0])
