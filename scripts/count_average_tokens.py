import json
import argparse
from xopen import xopen

def calculate_average_tokens(file_path):
    total_tokens = 0
    num_prompts = 0

    with xopen(file_path) as fin:  # Open gzip file
        for line in fin:  # Load the entire JSON file as a single JSON object
            data = json.loads(line)
            # Iterate over each question-prompt pair in the JSON object
            prompt = data.get('prompt')
            tokens = prompt.split()
            total_tokens += len(tokens)
            num_prompts += 1

    return total_tokens / num_prompts if num_prompts > 0 else 0

def main():
    parser = argparse.ArgumentParser(description='Calculate the average number of tokens per prompt in a JSONL.GZ file.')
    parser.add_argument('file_path', help='Path to the JSONL.GZ file')
    
    args = parser.parse_args()
    average_tokens = calculate_average_tokens(args.file_path)
    print(f"Average number of tokens per prompt: {average_tokens}")

if __name__ == "__main__":
    main()
