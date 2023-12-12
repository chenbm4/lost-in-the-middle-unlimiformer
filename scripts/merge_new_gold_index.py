import json
from xopen import xopen
from tqdm import tqdm

def load_data(file_path):
    data = {}
    with xopen(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            question = entry['question']
            data[question] = entry
    return data

def add_new_gold_index(source_file, index_file, output_file):
    # Load data from both files
    source_data = load_data(source_file)
    index_data = load_data(index_file)

    # Update source data with new_gold_index and prepare output data
    output_data = []
    for question, source_entry in tqdm(source_data.items()):
        index_entry = index_data.get(question)
        new_gold_index = index_entry.get('new_gold_index') if index_entry else None
        if new_gold_index is not None:
            output_entry = {
                "question": source_entry["question"],
                "model_answer": source_entry["model_answer"],
                "answers": source_entry["answers"],
                "new_gold_index": new_gold_index
            }
            output_data.append(output_entry)

    # Write updated data to output file
    with xopen(output_file, 'wt', encoding='utf-8') as f_out:
        for entry in output_data:
            f_out.write(json.dumps(entry) + '\n')

# File paths (update these paths as needed)
index_file = 'qa_prompts/50_total_documents/nq-open-randomized-prompts.jsonl.gz'
source_file = 'qa_predictions/50_total_documents/nq-open-randomized-uf-llama-predictions-20.jsonl.gz'
output_file = 'qa_predictions/50_total_documents/nq-open-randomized-uf-llama-predictions-20.jsonl.gz'

add_new_gold_index(source_file, index_file, output_file)
