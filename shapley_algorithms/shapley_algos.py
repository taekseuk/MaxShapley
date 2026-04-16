"""
shapley.py
----------

This module provides a class-based interface for computing Shapley values for information attribution in multi-source question answering tasks.

Each approach is encapsulated in a class that inherits from the abstract Shapley base class. These classes are designed to be instantiated with a dataset (list of source documents), and their compute methods are called with the necessary arguments to perform the attribution calculation.

- MaxShapley: Computes Shapley values using a max-based value function, where the value of a subset is determined by the maximum relevance of its sources to key points in the answer, as determined by an LLM pipeline.
- Baseline Shapley algorithms include: FullShapley (brute forcef), Monte Carlo with Uniform Sampling, Monte Carlo with Antithetic Sampling, and Leave One Out. 


"""
"""
shapley.py
----------

This module provides a class-based interface for computing Shapley values for information attribution in multi-source question answering tasks.

Shapley values are a principled way to fairly attribute the value of a prediction (e.g., an answer from an LLM) to the different information sources that contributed to it. This file provides two main approaches:

- MaxShapley: Computes Shapley values using a max-based value function, where the value of a subset is determined by the maximum relevance of its sources to key points in the answer, as determined by an LLM pipeline.

"""

from ast import List
import math, random, copy
from itertools import combinations, permutations
import random
import pandas as pd
import numpy as np
import logging, re, json

from llm_pipeline import create_llm_pipeline

class Shapley:
    """
    Abstract base class for Shapley value computation.

    This class is intended to be subclassed by specific Shapley value computation strategies.
    It stores the dataset (list of information sources) and defines the interface for compute().
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.input_tokens = 0
        self.output_tokens = 0

    def compute(self, *args, **kwargs):
        """
        Abstract method to compute Shapley values for the dataset.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement compute()")
    
    def normalize_scores(self, scores):
        if len(scores) > 0 and sum(scores) != 0:
            scores = [v / sum(scores) if v > 0.0 else 0.0 for v in scores]
        return scores

    def zero_out_tokens(self):
        self.input_tokens = 0
        self.output_tokens = 0

class MaxShapley(Shapley):
    """
    Computes Shapley values using a max-based value function and key point decomposition.
    """
    def compute(self, question, ground_truth, llm="anthropic"):
        logging.info("\n[MaxShapley]")
        information_sources = self.dataset
        llm_pipeline = create_llm_pipeline(llm)
        
        def run_llm_with_sources(source_docs, question, llm_pipeline, ground_truth):
            
            shuffled_source_docs = copy.deepcopy(source_docs)
            random.shuffle(shuffled_source_docs)
            
            # Separate keypoint generation
            response, in_tokens, out_tokens = llm_pipeline.run_task(
                "generate_response_with_info_subset",
                {
                    "question": question,
                    "sources": "\n\n".join(shuffled_source_docs)
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            keypoints, in_tokens, out_tokens = llm_pipeline.run_task(
                "separate_keypoints",
                {
                    "question": question,
                    "answer" : response
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            generalized_keypoints, in_tokens, out_tokens = llm_pipeline.run_task(
                "generalize_keypoints",
                {
                    "question": question,
                    "answer" : response,
                    "ground_truth": ground_truth,
                    "keypoints": keypoints['keypoints']
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            logging.info(f"Answer and justification: {response}")
            logging.info(f"Separate Keypoints: {keypoints}")
            logging.info(f"Generalized Keypoints: {generalized_keypoints}")

            return response, generalized_keypoints

        def compute_relevance_scores_on_the_fly(source, key_point, llm_pipeline, source_idx, keypoint_idx):
            if not key_point.strip():
                return 0.0
            logging.info(f"Source idx: {source_idx}, Keypoint: '{key_point}'")
            result, in_tokens, out_tokens = llm_pipeline.run_task(
                "keypoint_relevance_scoring",
                {
                    "keypoint": key_point,
                    "source": source
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens

            # Handle tuple result directly
            if isinstance(result, tuple) and len(result) == 2:
                score, explanation = result
                logging.info(f"Relevance score: {score}")
                logging.info(f"Relevance justification: {explanation}")
                return score

            assert isinstance(result, str), "result must be a string"

            # Fallback: try to parse as string (should not happen, but for robustness)
            try:
                explanation_match = re.search(r"EXPLANATION:\s*(.*?)(?:\n|SCORE:|$)", result, re.DOTALL)
                score_match = re.search(r"SCORE:\s*([0-9.]+)", result)
                explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
                score = float(score_match.group(1)) if score_match else 0.0

                logging.info(f"Relevance score: {score}")
                logging.info(f"Relevance justification: {explanation}")

                return score
            except Exception:
                logging.warning(f"Failed to parse relevance score from response: {result}")
                return 0.0

        def shapley_for_max_single_keypoint(x):
            n = len(x)
            phi = [1 for _ in range(n)]
            for i in range(n):
                phi[i] = x[i]/n
                for j in range(1, i):
                    p = 0.0
                    for k in range(2, j + 2):
                        p_A = 1.0 / n
                        p_B = (k - 1) / (n - 1)
                        p_C = 1.0
                        for l in range(1, n - j):
                            numerator = n - k - l + 1
                            denominator = n - l - 1
                            if denominator > 0:
                                p_C *= (numerator / denominator)
                            else:
                                p_C = 0.0
                                break
                        p += (p_A * p_B * p_C)
                    phi[i] += p * (x[i] - x[j])
            return phi

        def shapley_for_max(relevance_matrix):
            n_sources = relevance_matrix.shape[0]
            k_keypoints = relevance_matrix.shape[1]
            shapley_values = np.zeros(n_sources)
            for j in range(k_keypoints):
                scores = relevance_matrix[:, j]
                sorted_indices = np.argsort(scores)
                sorted_scores = scores[sorted_indices]
                sorted_phi = shapley_for_max_single_keypoint(sorted_scores.tolist())
                for orig_idx, sorted_idx in enumerate(sorted_indices):
                    shapley_values[sorted_idx] += sorted_phi[orig_idx]
            return shapley_values.tolist()

        logging.info(f"Sources: {list(range(len(information_sources)))}")
        result, generalized_result = run_llm_with_sources(information_sources, question, llm_pipeline, ground_truth)
        
        key_points = generalized_result['keypoints']
        if not key_points:
            logging.warning("No key points generated by LLM.")
            return [0.0] * len(information_sources)

        n_sources = len(information_sources)
        n_key_points = len(key_points)
        relevance_matrix = np.zeros((n_sources, n_key_points))
        for i, source in enumerate(information_sources):
            for j, key_point in enumerate(key_points):
                relevance_matrix[i, j] = compute_relevance_scores_on_the_fly(source, key_point, llm_pipeline, i, j)

        shapley_values = shapley_for_max(relevance_matrix)

        # Normalize before returning
        shapley_values = self.normalize_scores(shapley_values)
        logging.info(f"Normalized Shapley values: {shapley_values}")
        return shapley_values  

class FastMaxShapley(Shapley):
    """
    Optimized version of MaxShapley.
    Computes Shapley values using a max-based value function and key point decomposition.
    """
    def compute(self, question, ground_truth, llm="anthropic"):
        logging.info("\n[FastMaxShapley]")
        information_sources = self.dataset
        llm_pipeline = create_llm_pipeline(llm)
        
        def run_llm_with_sources(source_docs, question, llm_pipeline, ground_truth):
            
            shuffled_source_docs = copy.deepcopy(source_docs)
            random.shuffle(shuffled_source_docs)
            
            # Separate keypoint generation
            response, in_tokens, out_tokens = llm_pipeline.run_task(
                "generate_response_with_info_subset",
                {
                    "question": question,
                    "sources": "\n\n".join(shuffled_source_docs)
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            keypoints, in_tokens, out_tokens = llm_pipeline.run_task(
                "separate_keypoints",
                {
                    "question": question,
                    "answer" : response
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            generalized_keypoints, in_tokens, out_tokens = llm_pipeline.run_task(
                "generalize_keypoints",
                {
                    "question": question,
                    "answer" : response,
                    "ground_truth": ground_truth,
                    "keypoints": keypoints['keypoints']
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            logging.info(f"Answer and justification: {response}")
            logging.info(f"Separate Keypoints: {keypoints}")
            logging.info(f"Generalized Keypoints: {generalized_keypoints}")

            return response, generalized_keypoints

        def compute_relevance_scores_on_the_fly(source, key_point, llm_pipeline, source_idx, keypoint_idx):
            if not key_point.strip():
                return 0.0
            logging.info(f"Source idx: {source_idx}, Keypoint: '{key_point}'")
            result, in_tokens, out_tokens = llm_pipeline.run_task(
                "keypoint_relevance_scoring",
                {
                    "keypoint": key_point,
                    "source": source
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens

            # Handle tuple result directly
            if isinstance(result, tuple) and len(result) == 2:
                score, explanation = result
                logging.info(f"Relevance score: {score}")
                logging.info(f"Relevance justification: {explanation}")
                return score

            assert isinstance(result, str), "result must be a string"

            # Fallback: try to parse as string (should not happen, but for robustness)
            try:
                explanation_match = re.search(r"EXPLANATION:\s*(.*?)(?:\n|SCORE:|$)", result, re.DOTALL)
                score_match = re.search(r"SCORE:\s*([0-9.]+)", result)
                explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
                score = float(score_match.group(1)) if score_match else 0.0

                logging.info(f"Relevance score: {score}")
                logging.info(f"Relevance justification: {explanation}")

                return score
            except Exception:
                logging.warning(f"Failed to parse relevance score from response: {result}")
                return 0.0

        def shapley_for_max_single_keypoint(x):
            n = len(x)
            psum = 0.0
            phi = []
            for i in range(n):
                phi.append(x[i]/n + psum)
                if i+1<n:
                    psum += (x[i+1] - x[i]) * (1.0 / (n-(i+1)) - 1.0 / n)
            # Original MaxShapley implementation from the repository.
            # Note: it has an off-by-one bug in the inner loop.
            # Under 0-based indexing, `for j in range(1, i)` should be `for j in range(0, i)`.
            #
            # A quick sanity check is the case n = 2 with x[0] <= x[1].
            # For the max game, the correct Shapley values are:
            #   phi[0] = x[0] / 2
            #   phi[1] = x[1] - x[0] / 2
            # But the original loop gives phi[1] = x[1] / 2, since the j = 0 term is
            # skipped entirely. So the original implementation does not match the paper.

            # phi = [1 for _ in range(n)]
            # for i in range(n):
            #     phi[i] = x[i]/n
            #     for j in range(1, i):   However, it has an issue that loop should start from 0 instead of 1
            #         p = 0.0
            #         for k in range(2, j + 2):
            #             p_A = 1.0 / n
            #             p_B = (k - 1) / (n - 1)
            #             p_C = 1.0
            #             for l in range(1, n - j):
            #                 numerator = n - k - l + 1
            #                 denominator = n - l - 1
            #                 if denominator > 0:
            #                     p_C *= (numerator / denominator)
            #                 else:
            #                     p_C = 0.0
            #                     break
            #             p += (p_A * p_B * p_C)
            #         phi[i] += p * (x[i] - x[j])
            return phi

        def shapley_for_max(relevance_matrix):
            n_sources = relevance_matrix.shape[0]
            k_keypoints = relevance_matrix.shape[1]
            shapley_values = np.zeros(n_sources)
            for j in range(k_keypoints):
                scores = relevance_matrix[:, j]
                sorted_indices = np.argsort(scores)
                sorted_scores = scores[sorted_indices]
                sorted_phi = shapley_for_max_single_keypoint(sorted_scores.tolist())
                for orig_idx, sorted_idx in enumerate(sorted_indices):
                    shapley_values[sorted_idx] += sorted_phi[orig_idx]
            return shapley_values.tolist()

        logging.info(f"Sources: {list(range(len(information_sources)))}")
        result, generalized_result = run_llm_with_sources(information_sources, question, llm_pipeline, ground_truth)
        
        key_points = generalized_result['keypoints']
        if not key_points:
            logging.warning("No key points generated by LLM.")
            return [0.0] * len(information_sources)

        n_sources = len(information_sources)
        n_key_points = len(key_points)
        relevance_matrix = np.zeros((n_sources, n_key_points))
        for i, source in enumerate(information_sources):
            for j, key_point in enumerate(key_points):
                relevance_matrix[i, j] = compute_relevance_scores_on_the_fly(source, key_point, llm_pipeline, i, j)

        shapley_values = shapley_for_max(relevance_matrix)

        # Normalize before returning
        shapley_values = self.normalize_scores(shapley_values)
        logging.info(f"Normalized Shapley values: {shapley_values}")
        return shapley_values  
      
class FullShapley(Shapley):
    """
    Computes exact Shapley values using a brute-force approach with judge1 (0-1 scoring).

    For each possible subset of sources, this class uses an LLM to generate an answer
    and then scores the answer's quality using judge1 which provides a score between 0 and 1.
    The Shapley value for each source is computed by averaging its marginal contribution
    across all possible subsets. Final values are normalized to sum to 1.
    """
    
    def compute(self, question, ground_truth, llm="anthropic"):
        logging.info("\n[FullShapley]")
        source_docs = self.dataset
        n = len(source_docs)
        shapley_values = [0.0] * n
        llm_pipeline = create_llm_pipeline(llm)
        value_cache = {}
    
        def value_function(subset_indices):
            subset_key = tuple(sorted(subset_indices))
            if subset_key in value_cache:
                logging.debug(f"Cache hit for subset {subset_key}")
                return value_cache[subset_key]
            subset_sources = [source_docs[i] for i in subset_indices]
            
            # Shuffle the subset sources to randomize order
            shuffled_source_docs = copy.deepcopy(subset_sources)
            random.shuffle(shuffled_source_docs)
            
            logging.info(f"subset_indices: {subset_indices}")
            logging.info(f"Subset: {subset_key}")

            response, in_tokens, out_tokens = llm_pipeline.run_task(
                "generate_response_with_info_subset",
                {
                    "question": question,
                    "sources": "\n\n".join(shuffled_source_docs)
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            logging.info(f"LLM Response: {response}")

            judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                "judge1",
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "response": response
                }
            )
            if "No justification provided" in judge_result['justification']:
                logging.info(f"Trying judge again.")
                judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                    "judge1",
                    {
                        "question": question,
                        "ground_truth": ground_truth,
                        "response": response
                    }
                )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens

            logging.info(f"Full Judge Response: {judge_result}")

            score = judge_result['score']
            justification = judge_result['justification']
            logging.info(f"Score: {score}")
            logging.info(f"Judge Explanation: {justification}")

            value_cache[subset_key] = score
            return score

        for i in range(n):
            # affected_tmp = [0 for _ in range(n)]
            for j in range(n):
                subsets_without_i = [s for s in combinations(range(n), j) if i not in s]
                for subset in subsets_without_i:
                    subset_with_i = list(subset) + [i]
                    v_with = value_function(subset_with_i)
                    v_without = value_function(subset)
                    marginal_tmp = v_with - v_without
                    marginal = max(0, marginal_tmp)
                    shapley_values[i] += marginal / (math.comb(n-1, j) * n)

        shapley_values = self.normalize_scores(shapley_values)
        logging.info(f"Normalized Shapley values: {shapley_values}")
        return shapley_values

class MonteCarloUniform(Shapley):
    """
    Computes approximate Shapley values using a Monte Carlo approximation algorithm with uniform sampling.

    This method samples random permutations and computes marginal contributions along each permutation.
    It provides a good balance between computational efficiency and accuracy for large datasets.

    """
    
    def compute(self, question, ground_truth, llm="anthropic", m=1):
        logging.info("\n[MonteCarloUniform]")
        source_docs = self.dataset
        n = len(source_docs)
        shapley_values = [0] * n
        llm_pipeline = create_llm_pipeline(llm)
        value_cache = {}
    
        def value_function(subset_indices):
            subset_key = tuple(subset_indices)
            if subset_key in value_cache:
                logging.debug(f"Cache hit for subset {subset_key}")
                return value_cache[subset_key]
            subset_sources = [source_docs[i] for i in subset_indices]
            logging.info(f"Subset: {subset_indices}")

            response, in_tokens, out_tokens = llm_pipeline.run_task(
                "generate_response_with_info_subset",
                {
                    "question": question,
                    "sources": "\n\n".join(subset_sources)
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            logging.info(f"LLM Response: {response}")

            judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                "judge1",
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "response": response
                }
            )
            if "No justification provided" in judge_result['justification']:
                logging.info(f"Trying judge again.")
                judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                    "judge1",
                    {
                        "question": question,
                        "ground_truth": ground_truth,
                        "response": response
                    }
                )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens

            logging.info(f"Full Judge Response: {judge_result}")

            score = judge_result['score']
            justification = judge_result['justification']
            logging.info(f"Score: {score}")
            logging.info(f"Judge Explanation: {justification}")

            value_cache[subset_key] = score
            return score

        seen_perms = set() # to ensure no duplicate permutations
        subset = []
        empty_value = value_function(subset)
        for _ in range(m):
            done = False
            while not done:
                perm = [int(x) for x in np.random.permutation(n)]
                if tuple(perm) not in seen_perms:
                    seen_perms.add(tuple(perm))
                    done = True

            subset = []
            prev_value = empty_value

            for idx in perm:
                subset_with_i = subset + [idx]
                new_value = value_function(subset_with_i)
                marginal_contribution = new_value - prev_value
                shapley_values[idx] += max(0, marginal_contribution) / m
                prev_value = new_value
                subset = subset_with_i
            
            logging.info(f"Token consumption at: {self.input_tokens}, {self.output_tokens}")


        shapley_values = self.normalize_scores(shapley_values)
        logging.info(f"Normalized Shapley values: {shapley_values}")
        return shapley_values

class MonteCarloAntithetic(Shapley):
    """
    Computes approximate Shapley values using a Monte Carlo approximation algorithm with antithetic sampling.

    This method uses antithetic sampling by evaluating both forward and reverse permutations,
    which can reduce variance and improve convergence compared to uniform sampling.
    """
    
    def compute(self, question, ground_truth, llm="anthropic", m=16):
        logging.info("\n[MonteCarloAntithetic]")
        source_docs = self.dataset
        n = len(source_docs)
        shapley_values = [0.0] * n
        llm_pipeline = create_llm_pipeline(llm)
        value_cache = {}
    
        def value_function(subset_indices):
            subset_key = tuple(subset_indices)
            if subset_key in value_cache:
                logging.info(f"Cache hit for subset {subset_key}")
                return value_cache[subset_key]
            subset_sources = [source_docs[i] for i in subset_indices]
            logging.info(f"Subset: {subset_indices}")

            response, in_tokens, out_tokens = llm_pipeline.run_task(
                "generate_response_with_info_subset",
                {
                    "question": question,
                    "sources": "\n\n".join(subset_sources)
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            logging.info(f"LLM Response: {response}")

            judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                "judge1",
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "response": response
                }
            )
            if "No justification provided" in judge_result['justification']:
                logging.info(f"Trying judge again.")
                judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                    "judge1",
                    {
                        "question": question,
                        "ground_truth": ground_truth,
                        "response": response
                    }
                )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens

            logging.info(f"Full Judge Response: {judge_result}")

            score = judge_result['score']
            justification = judge_result['justification']
            logging.info(f"Score: {score}")
            logging.info(f"Judge Explanation: {justification}")

            value_cache[subset_key] = score
            return score

        seen_perms = set() # to ensure no duplicate permutations
        empty_value = value_function([])
        for _ in range(m):
            done = False
            while not done:
                perm = [int(x) for x in np.random.permutation(n)]
                reverse_perm = list(reversed(perm))
                if (tuple(perm) not in seen_perms) and (tuple(reverse_perm) not in seen_perms):
                    seen_perms.add(tuple(perm))
                    done = True

            # Forward pass
            subset = []
            prev_value = empty_value
            for i in range(n):
                subset_with_i = subset + [perm[i]]
                new_value = value_function(subset_with_i)
                marginal = max(0, new_value - prev_value)  # Custom clipping
                shapley_values[perm[i]] += marginal / m
                prev_value = new_value
                subset = subset_with_i

            # Reverse (antithetic) pass
            subset = []
            prev_value = empty_value
            for i in range(n):
                subset_with_i = subset + [reverse_perm[i]]
                new_value = value_function(subset_with_i)
                marginal = max(0, new_value - prev_value)
                shapley_values[reverse_perm[i]] += marginal / m
                prev_value = new_value
                subset = subset_with_i

            logging.info(f"Token consumption at: {self.input_tokens}, {self.output_tokens}")

        shapley_values = self.normalize_scores(shapley_values)
        logging.info(f"Normalized Shapley values: {shapley_values}")
        return shapley_values

class LeaveOneOut(Shapley):
    """
    Computes approximate Shapley values using Leave One Out algorithm approximation. 
    """
    def compute(self, question, ground_truth, llm="anthropic"):
        logging.info("\n[Leave-One-Out]")
        source_docs = self.dataset
        n = len(source_docs)
        shapley_values = [0.0] * n
        llm_pipeline = create_llm_pipeline(llm)

        def value_function(subset_indices):
            subset_sources = [source_docs[i] for i in subset_indices]
            logging.info(f"Subset: {subset_indices}")

            shuffled_source_docs = copy.deepcopy(subset_sources)
            random.shuffle(shuffled_source_docs)

            response, in_tokens, out_tokens = llm_pipeline.run_task(
                "generate_response_with_info_subset",
                {
                    "question": question,
                    "sources": "\n\n".join(shuffled_source_docs)
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            logging.info(f"LLM Response: {response}")

            judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                "judge1",
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "response": response
                }
            )
            if "No justification provided" in judge_result['justification']:
                logging.info(f"Trying judge again.")
                judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                    "judge1",
                    {
                        "question": question,
                        "ground_truth": ground_truth,
                        "response": response
                    }
                )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens

            logging.info(f"Full Judge Response: {judge_result}")

            score = judge_result['score']
            justification = judge_result['justification']
            logging.info(f"Score: {score}")
            logging.info(f"Judge Explanation: {justification}")
            return score

        utility_with_all = value_function([e for e in range(n)])

        for i in range(n):
            subset = [int(e) for e in range(n) if e != i]
            utility_without = value_function(subset)
            score = max(0, utility_with_all - utility_without) # Clip negatives as per your use case
            shapley_values[i] += score

        shapley_values = self.normalize_scores(shapley_values)
        logging.info(f"Normalized Shapley values: {shapley_values}")
        return shapley_values
