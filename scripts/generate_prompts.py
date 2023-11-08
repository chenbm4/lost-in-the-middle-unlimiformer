#!/usr/bin/env python3
"""Generate prompts for open-book QA from a data file with questions and retrieval results."""

import argparse
import json
import logging
import pathlib
import random
from tqdm import tqdm
from xopen import xopen

from lost_in_the_middle.prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
)

logger = logging.getLogger(__name__)

def generate_prompts(input_path, output_path, use_random_ordering, prompt_mention_random_ordering, query_aware_contextualization):
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    prompts = []

    # Fetch all of the prompts
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            question = input_example["question"]
            documents = [Document.from_dict(ctx) for ctx in input_example["ctxs"]]

            if use_random_ordering:
                # Randomly order only the distractors, keeping isgold documents at their existing index.
                original_gold_document = next((doc for doc in documents if doc.isgold), None)
                distractors = [doc for doc in documents if not doc.isgold]
                random.shuffle(distractors)
                if original_gold_document:
                    distractors.insert(0, original_gold_document)
                documents = distractors

            prompt = get_qa_prompt(
                question,
                documents,
                mention_random_ordering=prompt_mention_random_ordering,
                query_aware_contextualization=query_aware_contextualization,
            )
            prompts.append(prompt)

    # Save prompts to the output file
    with open(output_path, "w") as f:
        for prompt in prompts:
            f.write(prompt + "\n\n")

def get_qa_prompt(question, documents, mention_random_ordering, query_aware_contextualization):
    # Construct the prompt based on the parameters. This function will need to be implemented.
    # The following is a placeholder implementation.
    prompt_parts = [f"Question: {question}"]
    if mention_random_ordering:
        prompt_parts.append("Note: The documents are presented in random order.")
    for doc in documents:
        prompt_parts.append(f"Document Title: {doc.title}\n{doc.text}")
    if query_aware_contextualization:
        prompt_parts.append(f"Question: {question}")
    return "\n".join(prompt_parts)

# Define a minimal Document class for demonstration purposes.
class Document:
    def __init__(self, title, text, isgold=False):
        self.title = title
        self.text = text
        self.isgold = isgold

    @staticmethod
    def from_dict(ctx):
        return Document(ctx['title'], ctx['text'], ctx.get('isgold', False))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument("--output-path", help="Path to save generated prompts", required=True)
    parser.add_argument("--use-random-ordering", action="store_true", help="Randomize the ordering of the documents.")
    parser.add_argument("--prompt-mention-random-ordering", action="store_true", help="Mention random ordering in the prompt.")
    parser.add_argument("--query-aware-contextualization", action="store_true", help="Include the question before and after the documents in the prompt.")
    
    args = parser.parse_args()
    
    generate_prompts(
        args.input_path,
        args.output_path,
        args.use_random_ordering,
        args.prompt_mention_random_ordering,
        args.query_aware_contextualization
    )
