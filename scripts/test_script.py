import gzip
import json

def transfer_new_gold_index(original_file, destination_file, output_file):
    # Open the original file and read the new_gold_index values
    with gzip.open(original_file, 'rt', encoding='utf-8') as f:
        original_data = [json.loads(line) for line in f]

    # Open the destination file and read its content
    with gzip.open(destination_file, 'rt', encoding='utf-8') as f:
        destination_data = [json.loads(line) for line in f]

    # Check if both files have the same number of lines
    if len(original_data) != len(destination_data):
        raise ValueError("The original and destination files have different numbers of lines.")

    # Combine the data
    combined_data = []
    for orig, dest in zip(original_data, destination_data):
        new_entry = dest
        new_entry['new_gold_index'] = orig.get('new_gold_index')
        combined_data.append(new_entry)

    # Write the combined data to the output file
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        for entry in combined_data:
            f.write(json.dumps(entry) + '\n')

# Example usage
original_file = './qa_prompts/30_total_documents/nq-open-randomized-prompts.jsonl.gz'
destination_file = './qa_predictions/30_total_documents/nq-open-randomized-uf-llama-predictions.jsonl.gz'
output_file = './qa_predictions/30_total_documents/nq-open-randomized-uf-llama-predictions2.jsonl.gz'

transfer_new_gold_index(original_file, destination_file, output_file)
