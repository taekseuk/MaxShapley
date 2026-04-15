"""
calculate_shapley.py
-------------------
Driver script for computing Shapley values using multiple methods on multi-source QA datasets.

Usage:
python calculate_shapley.py
python calculate_shapley.py --dataset hotpot --index 5 --llm openai
"""

import argparse
import logging
import sys
import os
import csv
import time
import tempfile
from datetime import datetime
from typing import Set, Dict, Any
import json

from load_data import (
    load_hotpot_data_sample,
    load_msmarco_data_sample,
    load_musique_data_sample,
    MUSIQUE_DATA_PATH
)

from shapley_algorithms.shapley_algos import (
    FullShapley,
    MonteCarloUniform,
    MonteCarloAntithetic,
    LeaveOneOut,
    MaxShapley
)

from shapley_algorithms.kernel_shap import run_kernel_shap
# from llm_pipeline import OPENAI_MODEL, ANTHROPIC_MODEL


def normalize_scores(scores):
    if len(scores) > 0 and sum(scores) != 0:
        scores = [v / sum(scores) if v > 0.0 else 0.0 for v in scores]
    return scores


def parse_supporting_indices(supporting_str, context_parts, dataset_type='hotpot', index=0):
    """Parse supporting indices from the dataframe based on dataset type"""
    res = ""

    if dataset_type == 'msmarco':
        supporting_indices = set()
        for i, source in enumerate(context_parts):
            title = source[0]
            for fact_title, x in supporting_str:
                if fact_title == title and x >= 2:
                    supporting_indices.add(i)
                    break

        for i in supporting_indices:
            x = str(i + 1)
            if res == "" or not res:
                res += x
            else:
                res += ", " + x
        return res

    elif dataset_type == 'musique':
        example = load_musique_data_sample(index)
        all_context_parts = example["context"].split("\n\n")
        supporting_facts = example["supporting_facts"]
        supporting_indices = set()

        file_path = MUSIQUE_DATA_PATH
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        json_entry = data[index]

        for i, source in enumerate(all_context_parts):
            parts = source.split("\':\n", 1)
            if len(parts) != 2:
                continue
            title_part = parts[0]
            doc_context = parts[1]
            title = title_part.replace("Document \'", "").replace("\'", "'")

            for fact_title, x in supporting_facts:
                if fact_title == title:
                    for ctx in json_entry['context']:
                        if ctx[0] == fact_title and all(sent in doc_context for sent in ctx[1]):
                            supporting_indices.add(i)
                            break
                    break

        for i in supporting_indices:
            x = str(i + 1)
            if res == "" or not res:
                res += x
            else:
                res += ", " + x
        return res

    elif dataset_type == 'hotpot':
        example = load_hotpot_data_sample(index)
        all_context_parts = example["context"].split("\n\n")
        supporting_facts = example["supporting_facts"]
        supporting_indices = set()

        for i, source in enumerate(all_context_parts):
            parts = source.split("\':\n", 1)
            title_part = parts[0]
            doc_context = parts[1]
            title = title_part.replace("Document \'", "").replace("\'", "'")

            for fact_title, x in supporting_facts:
                if fact_title == title:
                    file_path = os.path.join(os.path.dirname(__file__), 'data/hotpotqa_annotated_subset.json')
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    idx = 0
                    json_entry = None
                    for entry in data:
                        if idx == index:
                            json_entry = entry
                            break
                        idx += 1

                    if json_entry:
                        for ctx in json_entry['context']:
                            if ctx[0] == fact_title and all(sent in doc_context for sent in ctx[1]):
                                supporting_indices.add(i)
                                break

        for i in supporting_indices:
            x = str(i + 1)
            if res == "" or not res:
                res += x
            else:
                res += ", " + x
        return res


def run_experiment(dataset, index, csv_path, llm, samples_u, samples_a, log_dir, timestamp, shapley_methods_to_run, rounds):
    # Load data
    example = None
    if dataset == 'hotpot':
        example = load_hotpot_data_sample(index, readable=False)
        dataset_and_index = f"HotPotQA Index: {index}"
    elif dataset == 'musique':
        example = load_musique_data_sample(index, readable=False)
        dataset_and_index = f"MuSiQUE Index: {index}"
    elif dataset == 'msmarco':
        example = load_msmarco_data_sample(index, readable=False)
        dataset_and_index = f"TREC MS MARCO Index: {index}"
    else:
        print(f"Unsupported --dataset arg (possible choices are 'hotpot', 'musique', 'msmarco'])")
        raise Exception

    # Extract sources
    try:
        sources = []
        for title, sentences in example['context']:
            doc_text = " ".join(sentences)
            sources.append(f"Document '{title}':\n{doc_text}")

        question = example['question']
        ground_truth = example.get('answer', '')
        num_sources = len(sources)

        print(f"Running {dataset_and_index}.\nQuery: {question}\nAnswer: {ground_truth}")
        print(f"Of the {num_sources} information sources: {parse_supporting_indices(example['supporting_facts'], example['context'], dataset_type=dataset, index=index)} are relevant.\n")

    except Exception:
        print(f"Problem loading data.")
        raise Exception

    # Prepare CSV headers
    shapley_methods = ['FullShapley', 'MaxShapley', 'MonteCarloUniform', 'MonteCarloAntithetic', 'LeaveOneOut']
    new_headers = ['index', 'llm_model', 'rounds']

    for method in shapley_methods:
        new_headers += [f'{method}_shapley_{i}' for i in range(num_sources)]
        new_headers += [f'{method}_execution_time', f'{method}_input_tokens', f'{method}_output_tokens']

    new_headers += ['MonteCarloUniform_sample_size', 'MonteCarloAntithetic_sample_size']

    # Handle existing CSV file or create new one
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', newline='') as fin:
                reader = csv.DictReader(fin)
                old_headers = reader.fieldnames or []
                final_headers = old_headers + [h for h in new_headers if h not in old_headers]

                # Atomic write with temp file
                dir_ = os.path.dirname(csv_path) or "."
                fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
                os.close(fd)
                try:
                    with open(tmp, 'w', newline='') as fout:
                        writer = csv.DictWriter(fout, fieldnames=final_headers, extrasaction="ignore", restval="")
                        writer.writeheader()
                        for row in reader:
                            writer.writerow(row)
                    os.replace(tmp, csv_path)
                finally:
                    if os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except OSError:
                            pass

        except Exception:
            print(f"Issue trying to open the output csv file. Check path.")
            raise Exception
    else:
        try:
            final_headers = list(new_headers)
            with open(csv_path, 'w', newline='') as fout:
                writer = csv.DictWriter(fout, fieldnames=final_headers)
                writer.writeheader()
        except Exception:
            print(f"Issue trying to open the output csv file.")
            raise Exception

    row = {
        'index': index,
        'llm_model': llm,
        'rounds': rounds,
        'MonteCarloUniform_sample_size': samples_u,
        'MonteCarloAntithetic_sample_size': samples_a
    }

    shapley_algorithms = {
        'FullShapley': FullShapley,
        'MaxShapley': MaxShapley,
        'MonteCarloUniform': MonteCarloUniform,
        'MonteCarloAntithetic': MonteCarloAntithetic,
        'LeaveOneOut': LeaveOneOut
    }

    for method_name in shapley_methods_to_run:
        if method_name == 'KernelSHAP':
            continue

        if method_name not in shapley_algorithms:
            continue

        start_time = time.time()
        
        
        algo = shapley_algorithms[method_name](sources)

        shap_avg = [0.0] * num_sources

        if method_name == 'MonteCarloUniform':
            values = algo.compute(question, ground_truth, llm=llm, m=samples_u)
        elif method_name == 'MonteCarloAntithetic':
            values = algo.compute(question, ground_truth, llm=llm, m=samples_a)
        else:
            values = algo.compute(question, ground_truth, llm=llm)

        if rounds == 1:
            shap_avg = values
        else:
            for j in range(num_sources):
                shap_avg[j] += values[j]

        elapsed = time.time() - start_time
        shap_avg = normalize_scores(shap_avg)

        for j in range(num_sources):
            row[f'{method_name}_shapley_{j}'] = shap_avg[j]

        row[f'{method_name}_execution_time'] = elapsed
        row[f'{method_name}_input_tokens'] = getattr(algo, 'input_tokens', 0)
        row[f'{method_name}_output_tokens'] = getattr(algo, 'output_tokens', 0)

    if 'KernelSHAP' in shapley_methods_to_run:
        start_time = time.time()
        kernel_scores, input_tokens, output_tokens = run_kernel_shap(
            question=question,
            answer=ground_truth,
            sources=sources,
            llm=llm,
            log_dir=log_dir
        )
        elapsed = time.time() - start_time
        kernel_scores = normalize_scores(kernel_scores)

        for j in range(num_sources):
            row[f'KernelSHAP_shapley_{j}'] = kernel_scores[j]

        row['KernelSHAP_execution_time'] = elapsed
        row['KernelSHAP_input_tokens'] = input_tokens
        row['KernelSHAP_output_tokens'] = output_tokens

        kernel_headers = [f'KernelSHAP_shapley_{i}' for i in range(num_sources)] + [
            'KernelSHAP_execution_time', 'KernelSHAP_input_tokens', 'KernelSHAP_output_tokens'
        ]
        for h in kernel_headers:
            if h not in final_headers:
                final_headers.append(h)

    # Append row
    with open(csv_path, 'a', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=final_headers, extrasaction="ignore", restval="")
        writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name: hotpot, musique, msmarco")
    parser.add_argument("--index", type=int, default=0, help="Index of the example to run")
    parser.add_argument("--llm", type=str, default="openai", help="LLM backend: openai or anthropic")
    parser.add_argument("--samples_u", type=int, default=100, help="Sample size for MonteCarloUniform")
    parser.add_argument("--samples_a", type=int, default=100, help="Sample size for MonteCarloAntithetic")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to store logs")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to output CSV")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds")
    parser.add_argument(
        "--shapley_methods",
        nargs="+",
        default=['FullShapley', 'MaxShapley', 'MonteCarloUniform', 'MonteCarloAntithetic', 'LeaveOneOut'],
        help="List of shapley methods to run"
    )

    args = parser.parse_args()

    llm = args.llm
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.dataset}_{args.index}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    if args.csv_path is None:
        csv_path = os.path.join(log_dir, f"{args.dataset}_{args.index}_results.csv")
    else:
        csv_path = args.csv_path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "run.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

    run_experiment(
        dataset=args.dataset,
        index=args.index,
        csv_path=csv_path,
        llm=llm,
        samples_u=args.samples_u,
        samples_a=args.samples_a,
        log_dir=log_dir,
        timestamp=timestamp,
        shapley_methods_to_run=args.shapley_methods,
        rounds=args.rounds
    )