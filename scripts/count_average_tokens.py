import json
import argparse
import gzip

def calculate_average_tokens(file_path):
    total_tokens = 0
    num_prompts = 0

    with gzip.open(file_path, 'rt', encoding='utf-8') as file:  # Open gzip file
        for line in file:
            try:
                data = json.loads(line)  # Load each line as a JSON object

                # Iterate over each question-prompt pair in the JSON object
                for question, details in data.items():
                    if 'prompt' in details:
                        prompt = details['prompt']
                        tokens = prompt.split()
                        total_tokens += len(tokens)
                        num_prompts += 1
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue

    return total_tokens / num_prompts if num_prompts > 0 else 0

def main():
    parser = argparse.ArgumentParser(description='Calculate the average number of tokens per prompt in a JSONL.GZ file.')
    parser.add_argument('file_path', help='Path to the JSONL.GZ file')
    
    args = parser.parse_args()
    average_tokens = calculate_average_tokens(args.file_path)
    print(f"Average number of tokens per prompt: {average_tokens}")

if __name__ == "__main__":
    main()