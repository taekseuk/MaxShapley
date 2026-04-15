from ast import Dict, List
import csv
import os
import json
import sys
import pandas as pd
from collections import defaultdict

MUSIQUE_DATA_PATH = os.environ.get("MUSIQUE_DATA_PATH", "data/musique_annotated_subset.json")


def load_hotpot_data_sample(index = 0, readable = True):
    """Load a specific example from the HotPotQA dev set."""
    path = "data/hotpotqa_annotated_subset.json"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"hootpotqa_annotated_subset.json not found at {path}.")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get the specified example
    example = data[index]

    if not readable:
        # return json format
        return example

    # Format context into readable text
    formatted_context = []
    for title, sentences in example["context"]:
        doc_text = " ".join(sentences)
        formatted_context.append(f"Document '{title}':\n{doc_text}")

    return {
        "question": example["question"],
        "context": "\n\n".join(formatted_context),
        "answer": example["answer"],  # Dev set has answers
        "supporting_facts": example["supporting_facts"]
    }


def load_msmarco_data_sample(index = 0, readable = True):
    """Load a specific example from the MS MARCO (TREC passages) annotated dataset."""
    path = "data/msmarco_annotated_subset.json"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ms_marco_annotated_subset.json not found at {path}.")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get the specified example
    example = data[index]

    if not readable:
        # return json format
        return example

    # Format context into readable text
    formatted_context = []
    for title, sentences in example["context"]:
        doc_text = " ".join(sentences)
        formatted_context.append(f"Passage '{title}':\n{doc_text}")

    return {
        "question": example["question"],
        "answer": "",
        "context": "\n\n".join(formatted_context),
        "supporting_facts": example["supporting_facts"]
    }


def load_musique_data_sample(index=0, readable=True):
    """
    Load a specific example from the Musique gold dataset.

    Args:
        index: Index of the example to load
        readable: If True, formats output for easy reading. If False, returns raw data.
    """
    path = MUSIQUE_DATA_PATH
    if not os.path.isfile(path):
        print(f"musique file not found at {path}.")
        raise FileNotFoundError()

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if index < 0 or index >= len(data):
        print(f"Index {index} out of range (0..{len(data)-1})")
        raise IndexError()

    example = data[index]

    if not readable:
        return example

    # Format context into readable text
    formatted_context = []
    for title, sentences in example["context"]:
        doc_text = " ".join(sentences)
        formatted_context.append(f"Document '{title}':\n{doc_text}")

    return {
        "question": example["question"],
        "answer": example["answer"],
        "context": "\n\n".join(formatted_context),
        "supporting_facts": example["supporting_facts"]
    }