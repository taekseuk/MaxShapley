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
import traceback

from load_data import (
    load_hotpot_data_sample,
    load_msmarco_data_sample,
    load_musique_data_sample
)

from shapley_algorithms.shapley_algos import (
    FullShapley,
    MonteCarloUniform,
    MonteCarloAntithetic,
    LeaveOneOut,
    MaxShapley,
    FastMaxShapley
)
from shapley_algorithms.kernel_shap import run_kernel_shap
from llm_pipeline import OPENAI_MODEL, ANTHROPIC_MODEL

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
        for i, source in enumerate(all_context_parts):
            parts = source.split("\':\n", 1)
            title_part = parts[0]
            doc_context = parts[1]
            title = title_part.replace("Document \'", "").replace("\'", "'")
            
            for fact_title, x in supporting_facts:
                if fact_title == title:

                    # Need to further check
                    file_path = os.path.join(os.path.dirname(__file__), 'data/musique_annotated_subset.json')
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
                            if ctx[0] == fact_title and all(x in doc_context for x in ctx[1]):
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
                            if ctx[0] == fact_title and all(x in doc_context for x in ctx[1]):
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
        print(f"Of the 6 information sources: {parse_supporting_indices(example['supporting_facts'], example['context'], dataset_type=dataset, index=index)} are relevant. ")
    except Exception as e:
        print("Problem loading data.")
        traceback.print_exc()
        raise Exception
    
    # Prepare CSV headers
    shapley_methods = ['FullShapley', 'MaxShapley', 'FastMaxShapley', 'MonteCarloUniform', 'MonteCarloAntithetic', 'LeaveOneOut']
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
            print(f"Issue trying to open the output csv file. Check path.")
            raise Exception
    
    # Initialize CSV row
    new_row = {h: "" for h in final_headers}
    new_row['index'] = index
    new_row['llm_model'] = OPENAI_MODEL if llm == 'openai' else ANTHROPIC_MODEL
    new_row['rounds'] = rounds
    new_row['MonteCarloUniform_sample_size'] = samples_u
    new_row['MonteCarloAntithetic_sample_size'] = samples_a
    
    # Run FullShapley
    if shapley_methods_to_run is None or any("FullShapley" == k for k in shapley_methods_to_run):
        try: 
            print("Running FullShapley")
            baseline_log = os.path.join(log_dir, f"{dataset}_FullShapley_{timestamp}.log")
            logging.basicConfig(filename=baseline_log, filemode='w', level=logging.INFO, format='%(message)s', force=True)
            logging.info(dataset_and_index)

            # Run rounds
            time_avg = 0
            input_token_avg = 0
            output_token_avg = 0
            shap_avg = [0, 0, 0, 0, 0, 0]

            for _ in range(rounds):
                start_time = time.time()
                baseline = FullShapley(sources)
                baseline_values = baseline.compute(
                    question=question,
                    ground_truth=ground_truth,
                    llm=llm
                )
                time_avg += (time.time() - start_time) / 3
                input_token_avg += (baseline.input_tokens / 3)
                output_token_avg += (baseline.output_tokens / 3)
                for k in range(len(baseline_values)):
                    shap_avg[k] += baseline_values[k]
        
            shap_avg = normalize_scores(shap_avg)
            
            new_row['FullShapley_execution_time'] = time_avg
            new_row['FullShapley_input_tokens'] = input_token_avg
            new_row['FullShapley_output_tokens'] = output_token_avg
            for i in range(num_sources):
                new_row[f'FullShapley_shapley_{i}'] = shap_avg[i]

            print(f"FullShapley logs saved to {baseline_log}")
            print(f"FullShapley execution time: {time_avg}")
            print(f"FullShapley shapley values: {shap_avg}\n")
            for k in range(len(shap_avg)):
                print(f"Source {k} shapley value: {shap_avg[k]}")
            print()
        except Exception:
            print(f"Issue running FullShapley.\n")
    
    # Run MaxShapley
    if shapley_methods_to_run is None or any("MaxShapley" == k for k in shapley_methods_to_run):
        try:
            print("Running MaxShapley")
            maxshapley_log = os.path.join(log_dir, f"{dataset}_MaxShapley_{timestamp}.log")
            logging.basicConfig(filename=maxshapley_log, filemode='w', level=logging.INFO, format='%(message)s', force=True)
            logging.info(dataset_and_index)
            
            # Run rounds
            time_avg = 0
            input_token_avg = 0
            output_token_avg = 0
            shap_avg = [0, 0, 0, 0, 0, 0]

            for _ in range(rounds):
                start_time = time.time()
                maxshap = MaxShapley(sources)
                max_values = maxshap.compute(
                    question=question,
                    ground_truth=ground_truth,
                    llm=llm
                )
                time_avg += (time.time() - start_time) / 3
                input_token_avg += (maxshap.input_tokens / 3)
                output_token_avg += (maxshap.output_tokens / 3)
                for k in range(len(max_values)):
                    shap_avg[k] += max_values[k]

            shap_avg = normalize_scores(shap_avg)
            
            new_row['MaxShapley_execution_time'] = time_avg
            new_row['MaxShapley_input_tokens'] = input_token_avg
            new_row['MaxShapley_output_tokens'] = output_token_avg
            for i in range(num_sources):
                new_row[f'MaxShapley_shapley_{i}'] = shap_avg[i]

            print(f"MaxShapley logs saved to {maxshapley_log}")
            print(f"MaxShapley execution time: {time_avg}")
            for k in range(len(shap_avg)):
                print(f"Source {k} shapley value: {shap_avg[k]}")
            print()
        except Exception:
            print(f"Issue running MaxShapley.\n")

    # Run FastMaxShapley
    if shapley_methods_to_run is None or any("FastMaxShapley" == k for k in shapley_methods_to_run):
        try:
            print("Running FastMaxShapley")
            fastmaxshapley_log = os.path.join(log_dir, f"{dataset}_FastMaxShapley_{timestamp}.log")
            logging.basicConfig(filename=fastmaxshapley_log, filemode='w', level=logging.INFO, format='%(message)s', force=True)
            logging.info(dataset_and_index)
            
            # Run rounds
            time_avg = 0
            input_token_avg = 0
            output_token_avg = 0
            shap_avg = [0, 0, 0, 0, 0, 0]

            for _ in range(rounds):
                start_time = time.time()
                fastmaxshap = FastMaxShapley(sources)
                fastmax_values = fastmaxshap.compute(
                    question=question,
                    ground_truth=ground_truth,
                    llm=llm
                )
                time_avg += (time.time() - start_time) / 3
                input_token_avg += (fastmaxshap.input_tokens / 3)
                output_token_avg += (fastmaxshap.output_tokens / 3)
                for k in range(len(fastmax_values)):
                    shap_avg[k] += fastmax_values[k]

            shap_avg = normalize_scores(shap_avg)
            
            new_row['FastMaxShapley_execution_time'] = time_avg
            new_row['FastMaxShapley_input_tokens'] = input_token_avg
            new_row['FastMaxShapley_output_tokens'] = output_token_avg
            for i in range(num_sources):
                new_row[f'FastMaxShapley_shapley_{i}'] = shap_avg[i]

            print(f"FastMaxShapley logs saved to {fastmaxshapley_log}")
            print(f"FastMaxShapley execution time: {time_avg}")
            for k in range(len(shap_avg)):
                print(f"Source {k} shapley value: {shap_avg[k]}")
            print()
        except Exception:
            print(f"Issue running FastMaxShapley.\n")

    # Run MonteCarloUniform
    if shapley_methods_to_run is None or any("MonteCarloUniform" == k for k in shapley_methods_to_run):
        try:
            print("Running MonteCarloUniform")
            mcu_log = os.path.join(log_dir, f"{dataset}_MonteCarloUniform_{timestamp}.log")
            logging.basicConfig(filename=mcu_log, filemode='w', level=logging.INFO, format='%(message)s', force=True)
            logging.info(dataset_and_index)

            # Run rounds
            time_avg = 0
            input_token_avg = 0
            output_token_avg = 0
            shap_avg = [0, 0, 0, 0, 0, 0]

            for _ in range(rounds):
                start_time = time.time()
                mc_uniform = MonteCarloUniform(sources)
                mcu_values = mc_uniform.compute(
                    question=question,
                    ground_truth=ground_truth,
                    llm=llm,
                    m=samples_u
                )
                time_avg += (time.time() - start_time) / 3
                input_token_avg += (mc_uniform.input_tokens / 3)
                output_token_avg += (mc_uniform.output_tokens / 3)
                for k in range(len(mcu_values)):
                    shap_avg[k] += mcu_values[k]

            shap_avg = normalize_scores(shap_avg)
            
            new_row['MonteCarloUniform_execution_time'] = time_avg
            new_row['MonteCarloUniform_input_tokens'] = input_token_avg
            new_row['MonteCarloUniform_output_tokens'] = output_token_avg
            for i in range(num_sources):
                new_row[f'MonteCarloUniform_shapley_{i}'] = shap_avg[i]

            print(f"MonteCarloUniform logs saved to {mcu_log}")
            print(f"MonteCarloUniform execution time: {time_avg}")
            for k in range(len(shap_avg)):
                print(f"Source {k} shapley value: {shap_avg[k]}")
            print()
        except Exception:
            print(f"Issue running MonteCarloUniform.\n")
        
    # Run MonteCarloAntithetic
    if shapley_methods_to_run is None or any("MonteCarloAntithetic" == k for k in shapley_methods_to_run):
        try:
            print("Running MonteCarloAntithetic")
            mca_log = os.path.join(log_dir, f"{dataset}_MonteCarloAntithetic_{timestamp}.log")
            logging.basicConfig(filename=mca_log, filemode='w', level=logging.INFO, format='%(message)s', force=True)
            logging.info(dataset_and_index)
            
            # Run rounds
            time_avg = 0
            input_token_avg = 0
            output_token_avg = 0
            shap_avg = [0, 0, 0, 0, 0, 0]

            for _ in range(rounds):
                start_time = time.time()
                mc_antithetic = MonteCarloAntithetic(sources)
                mca_values = mc_antithetic.compute(
                    question=question,
                    ground_truth=ground_truth,
                    llm=llm,
                    m=samples_a
                )
                time_avg += (time.time() - start_time) / 3
                input_token_avg += (mc_antithetic.input_tokens / 3)
                output_token_avg += (mc_antithetic.output_tokens / 3)
                for k in range(len(mca_values)):
                    shap_avg[k] += mca_values[k] 

            shap_avg = normalize_scores(shap_avg)
            
            new_row['MonteCarloAntithetic_execution_time'] = time_avg
            new_row['MonteCarloAntithetic_input_tokens'] = input_token_avg
            new_row['MonteCarloAntithetic_output_tokens'] = output_token_avg
            for i in range(num_sources):
                new_row[f'MonteCarloAntithetic_shapley_{i}'] = shap_avg[i]

            print(f"MonteCarloAntithetic logs saved to {mca_log}")
            print(f"MonteCarloAntithetic execution time: {time_avg}")
            for k in range(len(shap_avg)):
                print(f"Source {k} shapley value: {shap_avg[k]}")
            print()
        except Exception:
            print(f"Issue running MonteCarloAntithetic.\n")
    
    # Run LeaveOneOut
    if shapley_methods_to_run is None or any("LeaveOneOut" == k for k in shapley_methods_to_run):
        try:
            print("Running LeaveOneOut")
            loo_log = os.path.join(log_dir, f"{dataset}_LeaveOneOut_{timestamp}.log")
            logging.basicConfig(filename=loo_log, filemode='w', level=logging.INFO, format='%(message)s', force=True)
            logging.info(dataset_and_index)

            # Run rounds
            time_avg = 0
            input_token_avg = 0
            output_token_avg = 0
            shap_avg = [0, 0, 0, 0, 0, 0]

            for _ in range(rounds):
                start_time = time.time()
                loo = LeaveOneOut(sources)
                loo_values = loo.compute(
                    question=question,
                    ground_truth=ground_truth,
                    llm=llm
                )
                time_avg += (time.time() - start_time) / 3
                input_token_avg += (loo.input_tokens / 3)
                output_token_avg += (loo.output_tokens / 3)
                for k in range(len(loo_values)):
                    shap_avg[k] += loo_values[k]

            shap_avg = normalize_scores(shap_avg)
            
            new_row['LeaveOneOut_execution_time'] = time_avg
            new_row['LeaveOneOut_input_tokens'] = input_token_avg
            new_row['LeaveOneOut_output_tokens'] = output_token_avg
            for i in range(num_sources):
                new_row[f'LeaveOneOut_shapley_{i}'] = shap_avg[i]
            
            print(f"LeaveOneOut logs saved to {loo_log}")
            print(f"LeaveOneOut execution time: {time_avg}")
            for k in range(len(shap_avg)):
                print(f"Source {k} shapley value: {shap_avg[k]}")
            print()
        except Exception:
            print(f"Issue running LeaveOneOut.\n")
    
    # Write row to CSV
    try:
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=final_headers)
            writer.writerow(new_row)
    except Exception:
        print(f"Issue opening or saving csv output file. Check path.")
        raise Exception
    
    # Run KernelSHAP (uses MonteCarloUniform log data)   
    if shapley_methods_to_run is None or (any("KernelSHAP" == k for k in shapley_methods_to_run) and any("MonteCarloUniform" == k for k in shapley_methods_to_run)):
        try:
            print("Starting KernelSHAP")
            kernel_vals = run_kernel_shap(
                log_file=mcu_log,  # Use MonteCarloUniform log file
                source_csv=csv_path,
                output=csv_path,
                alpha=0.001,
                permutations=samples_u
            )
            for k in range(len(shap_avg)):
                print(f"Source {k} shapley value: {kernel_vals[k]}")
            print()
        except Exception as e:
            print(f"KernelSHAP failed: {e}\n")
        
    print(f"\nResults saved to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Shapley value experiments")
    parser.add_argument('--dataset', type=str, default='musique', choices=['hotpot', 'musique', 'msmarco'])
    parser.add_argument('--index', type=int, default=0, help='Index of data sample (default: 0)')
    parser.add_argument('--llm', type=str, default='openai', choices=['anthropic', 'openai'])
    parser.add_argument('--log', type=str, default='logs/', help='Path to log file or directory (default: logs/)')
    parser.add_argument('--csv', type=str, default='results/', help='Path to CSV file or directory (default: results/)')
    parser.add_argument('--rounds', type=int, default=3, help='Number of rounds to run.')
    parser.add_argument('--samples_u', type=int, default=1, help='Samples for MonteCarloUniform (default: 1)')
    parser.add_argument('--samples_a', type=int, default=1, help='Samples for MonteCarloAntithetic (default: 1)')
    parser.add_argument('--shapley_methods', nargs='+', type=str, default=None, help='Which Shapley implementations to run (e.g. FullShapley, MaxShapley, FastMaxShapley, MonteCarloUniform, MonteCarloAntithetic, KernelSHAP, LeaveOneOut)')
    
    args = parser.parse_args()
    
    # Configure logging - will create separate log files for each method
    log_dir = args.log
    if not log_dir.endswith('/'):
        log_dir = os.path.dirname(log_dir) or 'logs/'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure CSV output
    try: 
        csv_path = args.csv
        if os.path.isdir(csv_path) or csv_path.endswith('/'):
            os.makedirs(csv_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(csv_path, f"{args.dataset}_{timestamp}.csv")
        else:
            csv_dir = os.path.dirname(csv_path)
            if csv_dir:
                os.makedirs(csv_dir, exist_ok=True)
    except Exception: 
        print(f"Issue configuring CSV output. Check path.")
        return

    try: 
        for idx in range(args.index + 1):
            run_experiment(args.dataset, idx, csv_path, args.llm, args.samples_u, args.samples_a, log_dir, timestamp, args.shapley_methods, args.rounds)
    except Exception:
        print(f"Problem running experiments.")
        return

    

if __name__ == "__main__":
    main()
