# !/usr/bin/env python3
# for gold_index in 0 4 9; do
#     python -u ./scripts/get_qa_responses_from_unlimiformer.py \
#         --input-path qa_data/10_total_documents/nq-open-10_total_documents_gold_at_${gold_index}.jsonl.gz \
#         --num-gpus 1 \
#         --max-new-tokens 100 \
#         --batch-size 1 \
#         --max-memory-per-gpu 32 \
#         --num-gpus 1 \
#         --model TheBloke/Llama-2-7B-chat-GPTQ  \
#         --output-path qa_predictions/10_total_documents/nq-open-10_total_documents_gold_at_${gold_index}-unlimiformer-llama-2-7b-chat-gptq-predictions.jsonl.gz
# done
"""Given a data file with questions and retrieval results to use, run GPT2 to get responses.

Currently supports `gpt2-xl`.

The retrieval results are used in the exact order that they're given.
"""
import argparse
import json
import logging
import pathlib
import random
import sys
import os
from copy import deepcopy

from tqdm import tqdm
from xopen import xopen

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(src_dir)

from lost_in_the_middle.prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
)

logger = logging.getLogger(__name__)
random.seed(0)

# python src/run_generation.py 
# --model_type llama 
# --model_name_or_path TheBloke/Llama-2-7B-chat-GPTQ 
# --prefix "<s>[INST] <<SYS>>\n You are a helpful assistant. Answer with detailed responses according 
# to the entire instruction or question. \n<</SYS>>\n\n Summarize the following book: "  
# --prompt example_inputs/harry_potter.txt 
# --suffix " [/INST]" 
# --test_unlimiformer 
# --length 200 
# --layer_begin 16 
# --use_datastore False


def main(
    input_path,
    model_name,
    closedbook,
    prompt_mention_random_ordering,
    use_random_ordering,
    use_all_random_ordering,
    query_aware_contextualization,
    output_path,
):
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    examples = []
    prompts = {}
    all_model_documents = []
    did_format_warn = False

    # Fetch all of the prompts
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            # Get the prediction for the input example
            question = input_example["question"]
            answers = input_example["answers"]
            if closedbook:
                documents = []
            else:
                documents = []
                for ctx in deepcopy(input_example["ctxs"]):
                    documents.append(Document.from_dict(ctx))
                if not documents:
                    raise ValueError(f"Did not find any documents for example: {input_example}")

            if use_random_ordering:
                # Randomly order only the distractors (isgold is False), keeping isgold documents
                # at their existing index.
                (original_gold_index,) = [idx for idx, doc in enumerate(documents) if doc.isgold is True]
                original_gold_document = documents[original_gold_index]
                distractors = [doc for doc in documents if doc.isgold is False]
                random.shuffle(distractors)
                distractors.insert(original_gold_index, original_gold_document)
                documents = distractors
            
            if use_all_random_ordering:
                # Randomly order all documents
                original_gold_index = next((idx for idx, doc in enumerate(documents) if doc.isgold), None)
                random.shuffle(documents)
                new_gold_index = next((idx for idx, doc in enumerate(documents) if doc.isgold), None)

            if closedbook:
                prompt = get_closedbook_qa_prompt(question)
            else:
                prompt = get_qa_prompt(
                    question,
                    documents,
                    mention_random_ordering=prompt_mention_random_ordering,
                    query_aware_contextualization=query_aware_contextualization,
                )

            if "instruct" in model_name:
                if did_format_warn is False:
                    logger.warning(f"Model {model_name} appears to be an instruct model, applying instruct formatting")
                    did_format_warn = True
                prompt = format_instruct_prompt(prompt)
            elif "chat" in model_name:
                if did_format_warn is False:
                    logger.warning(f"Model {model_name} appears to be a chat model, applying chat formatting")
                    did_format_warn = True
                prompt = format_chat_prompt(prompt)
            prompts[question] = {
                "question": question,
                "prompt": prompt,
                "answers": answers,
                "new_gold_index": new_gold_index if use_all_random_ordering else None
            }
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)

    with xopen(output_path, "w") as f:
        for prompt in prompts.values():
            f.write(json.dumps(prompt) + "\n")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def format_instruct_prompt(instruction):
    INSTRUCTION_KEY = "### Instruction:"
    RESPONSE_KEY = "### Response:"
    INTRO_BLURB = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    )
    PROMPT_FOR_GENERATION = "{intro}\n{instruction_key}\n{instruction}\n{response_key}\n".format(
        intro=INTRO_BLURB,
        instruction_key=INSTRUCTION_KEY,
        instruction=instruction,
        response_key=RESPONSE_KEY,
    )
    return PROMPT_FOR_GENERATION

def format_chat_prompt(instruction):
    # Format the prompt according to LLaMa 2 chat model requirements
    prompt = f"<s>[INST] <<SYS>>\n Below is an instruction that describes a task." \
        f"Write a response that appropriately completes the request. \n<</SYS>>\n\n{instruction}"
    return prompt

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument(
        "--model",
        help="Model to use in generating responses",
        required=True,
        choices=["gpt2-xl", "facebook/opt-125m", "meta-llama/Llama-2-7b-chat-hf"],
    )
    parser.add_argument(
        "--closedbook", action="store_true", help="Run the model in closed-book mode (i.e., don't use documents)."
    )
    parser.add_argument(
        "--prompt-mention-random-ordering",
        action="store_true",
        help="Mention that search results are ordered randomly in the prompt",
    )
    parser.add_argument(
        "--use-random-ordering",
        action="store_true",
        help="Randomize the ordering of the distractors, rather than sorting by relevance.",
    )
    parser.add_argument(
        "--use-all-random-ordering",
        action="store_true",
        help="Randomize the ordering of all documents, rather than sorting by relevance.",
    )
    parser.add_argument(
        "--query-aware-contextualization",
        action="store_true",
        help="Place the question both before and after the documents.",
    )
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model,
        args.closedbook,
        args.prompt_mention_random_ordering,
        args.use_random_ordering,
        args.use_all_random_ordering,
        args.query_aware_contextualization,
        args.output_path,
    )
    logger.info("finished running %s", sys.argv[0])
