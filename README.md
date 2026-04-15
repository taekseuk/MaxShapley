# MaxShapley: Information Attribution via Shapley Values

Official implementation of MaxShapley from **[MaxShapley: Towards Incentive-compatible Generative Search with Fair Context Attribution]**.

## Overview

This repository computes Shapley values to fairly attribute the contribution of information sources in multi-source question answering. We implement multiple Shapley value computation methods including our novel **MaxShapley** approach.

## Installation

```bash
pip install -r requirements.txt
```

Set your API keys:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

## Quick Start

Run with defaults: MuSiQue dataset, index 0, OpenAI, all baselines (see below)
```bash
python calculate_shapley.py
```

Run with custom parameters:
```bash
python calculate_shapley.py --dataset hotpot --index 5 --llm openai --shapley_methods MaxShapley
```

## Performance Optimization 🚀

### Batched Scoring (Production-Ready)

We provide an optimized version of MaxShapley with **batched relevance scoring** that achieves:
- **1.85x faster** execution
- **44% fewer tokens** (lower cost)
- **11% quality drop** (acceptable for production)

**Compare implementations:**
```bash
# Set OpenRouter API key (or use demo/.env)
export OPENROUTER_API_KEY="your-key"

# Run comparison on 3 MuSiQUE samples
./run_comparison.sh 3
```

**Results**: See `BATCHED_OPTIMIZATION_RESULTS.md` for detailed analysis.

**When to use**:
- **Batched** (`maxshapley_batched.py`): Production deployments, real-time inference
- **Original** (`shapley_algos.py`): Research evaluation, benchmarking, maximum accuracy

## Methods Implemented

- **MaxShapley**: (Ours) Novel max-based approach with key point decomposition
- **BaselineShapley**: Exact Shapley values (evaluates all coalitions, computationally expensive)
- **MonteCarloUniform**: Monte Carlo approximation of BaselineShapley using uniform sampling
- **MonteCarloAntithetic**: Monte Carlo approximation of BaselineShapley using antithetic sampling for variance reduction
- **LeaveOneOut**: Approximation of BaselineShapley using leave-one-out evaluation
- **KernelSHAP**: Linear regression-based approximation (uses MonteCarloUniform logs, so must be run with MonteCarloUniform)

## Command Line Arguments

```bash
python calculate_shapley.py # Will use default parameters
python calculate_shapley.py --dataset hotpot --shapley_methods FullShapley --index 10 
```
**Options:**
- `--dataset`: Which Shapley implementations to run (e.g. FullShapley, MaxShapley MonteCarloUniform, MonteCarloAntithetic, KernelSHAP, LeaveOneOut) [default: `all`]
- `--shapley_methods`: Number of samples for MonteCarloAntithetic [default: `16`]
- `--index`: Data sample index (0-29) [default: `0`]
- `--llm`: LLM provider (`anthropic`, `openai`) [default: `anthropic`]
- `--log`: Directory for log files [default: `logs/`]
- `--csv`: Directory for CSV results [default: `results/`]
- `--rounds`: Number of rounds [default: `3`]
- `--samples_u`: Number of samples for MonteCarloUniform [default: `1`]
- `--samples_a`: Number of samples for MonteCarloAntithetic [default: `1`]

## Output

The script generates:

**Console**
Each data sample's query, answer, and an indication of which sources are supporting. 
Each shapley algorithm's execution time and shapley values (in a list) averaged across rounds. 

**CSV file** (`results/` directory):
Each row corresponds to a single example (index). 
Shapley values for the six information sources appear in columns named {shapley_method}\_shapley\_{source_index}, where source_index is an integer from 0 to 5.
If a shapley algorithm did not run, the column still exists but the value is blank. 

**Log files** (`logs/` directory):
- Separate log file for each method
- Contains detailed execution traces and intermediate results

## Datasets

We release re-annotated subsets of HotPotQA, MuSiQue, and MS-MARCO:
- `hotpotqa_annotated_subset.json`
- `musique_annotated_subset.json`
- `msmarco_annotated_subset.json`

Each contains examples with a query, information sources and their titles, and an indication of which information sources are relevant to answering the query. 
HotPotQA & MuSiQue: supporting_facts field contains tuples of (title, _) where the title indicates relevant sources
MS-MARCO: supporting_facts field contains tuples of (title, relevance) where relevance is 0 (not relevant) or above 0 (relevant)

Each sample contains:
- `question`: The query
- `context`: List of (title, sentences) tuples
- `answer`: Ground truth answer
- `supporting_facts`: Relevant sources

## Citation
If you found this code useful for a publication, please consider citing our paper: 

@article{maxshapley,  
&nbsp;&nbsp;&nbsp;&nbsp;author = {Patel, Sara and Zhou, Mingxun and Fanti, Giulia},  
&nbsp;&nbsp;&nbsp;&nbsp;title = {MaxShapley: Towards Incentive-compatible Generative Search with Fair Context Attribution},  
&nbsp;&nbsp;&nbsp;&nbsp;month = {November},  
&nbsp;&nbsp;&nbsp;&nbsp;publisher = {arXiv},  
&nbsp;&nbsp;&nbsp;&nbsp;year = 2025  
}
