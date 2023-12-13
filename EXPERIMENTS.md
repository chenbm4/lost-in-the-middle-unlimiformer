# Multi-Document Question Answering

Note: all of these experiments were run on one A100 GPUs with 80GB of
VRAM. You may need to modify commands to fit your own computing environment
(e.g., changing the batch size, the max memory per GPU, the number of GPUs, etc)

If using Unlimiformer, enable the --test_unlimiformer flag. Otherwise, leave it out to run using the base model. The input will be truncated to fit within the maximum context length.

### LLaMa 2 7B Chat HF on oracle

Generating prompts:

```
python ./scripts/generate_simplified_prompts.py \
	--input-path qa_data/nq-open-oracle.jsonl.gz \
	--model meta-llama/Llama-2-7b-chat-hf \
	--output-path qa_prompts/open_oracle/simplified_prompts.jsonl.gz
```

Getting predictions:

```
python ./unlimiformer/src/run_generation_json.py \
    --model_type llama \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --input_file qa_prompts/open_oracle/simplified_prompts.jsonl.gz \
    --output_file ./qa_predictions/open_oracle/llama-predictions.jsonl.gz \
    --suffix " [/INST]" \
    --fp16 \
    --length 100 \
    --layer_begin 16 \
    --use_datastore False
```

Evaluating:

```
python -u ./scripts/evaluate_qa_responses.py \
    --input-path ./qa_predictions/open_oracle/llama-predictions.jsonl.gz \
    --output-path ./qa_predictions/open_oracle/llama-predictions-scored.jsonl.gz
```

### LLaMa 2 7B Chat HF on closed-book

Generating prompts:

```
python ./scripts/generate_simplified_prompts.py \
	--input-path qa_data/nq-open-oracle.jsonl.gz \
	--model meta-llama/Llama-2-7b-chat-hf \
    --closedbook \
	--output-path qa_prompts/closed_book/simplified_prompts.jsonl.gz
```

Getting predictions:

```
python ./unlimiformer/src/run_generation_json.py \
    --model_type llama \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --input_file qa_prompts/closed_book/simplified_prompts.jsonl.gz \
    --output_file ./qa_predictions/closed_book/llama-predictions.jsonl.gz \
    --suffix " [/INST]" \
    --fp16 \
    --length 100 \
    --layer_begin 16 \
    --use_datastore False
```

Evaluating:

```
python -u ./scripts/evaluate_qa_responses.py \
    --input-path ./qa_predictions/closed_book/llama-predictions.jsonl.gz \
    --output-path ./qa_predictions/closed_book/llama-predictions-scored.jsonl.gz
```

### LLaMa 2 7B Chat HF on 20 document setting

Generating prompts:

```
for gold_index in 0 4 9 14 19; do \
python ./scripts/generate_simplified_prompts.py \
	--input-path qa_data/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}.jsonl.gz \
	--model meta-llama/Llama-2-7b-chat-hf \
	--output-path qa_prompts/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-simplified.jsonl.gz; \
done
```

Getting predictions:

```
for gold_index in 0 4 9 14 19; do \
python ./unlimiformer/src/run_generation_json.py \
    --model_type llama \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --input_file qa_prompts/20_total_documents/nq-open-20_total_documents_gold_at_${gold_index}-simplified.jsonl.gz \
    --output_file ./qa_predictions/20_total_documents/nq-open-gold_at_${gold_index}-uf-llama-predictions.jsonl.gz \
    --suffix " [/INST]" \
    --test_unlimiformer \
    --fp16 \
    --length 100 \
    --layer_begin 16 \
    --use_datastore False; \
done
```

Evaluating:

```
for gold_index in 0 4 9 14 19; do \
python -u ./scripts/evaluate_qa_responses.py \
    --input-path ./qa_predictions/20_total_documents/nq-open-gold_at_${gold_index}-uf-llama-predictions.jsonl.gz \
    --output-path ./qa_predictions/20_total_documents/nq-open-gold_at_${gold_index}-uf-llama-predictions-scored.jsonl.gz; \
done
```

### LLaMa 2 7B Chat HF on 50 document setting with randomized document location

Generating prompts:

```
python ./scripts/generate_simplified_prompts.py `
    --input-path qa_data/50_total_documents/nq-open-50_total_documents.jsonl.gz `
    --model meta-llama/Llama-2-7b-chat-hf `
    --output-path qa_prompts/50_total_documents/nq-open-randomized-prompts.jsonl.gz `
	--use-all-random-ordering
```

Getting predictions:

```
python ./unlimiformer/src/run_generation_json.py \
    --model_type llama \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --input_file qa_prompts/50_total_documents/nq-open-randomized-prompts.jsonl.gz \
    --output_file ./qa_predictions/50_total_documents/nq-open-randomized-uf-llama-predictions.jsonl.gz \
    --suffix " [/INST]" \
    --test_unlimiformer \
    --fp16 \
    --length 100 \
    --layer_begin 16 \
    --use_datastore False
```

Evaluating:

```
python -u ./scripts/evaluate_qa_responses.py \
    --input-path ./qa_predictions/50_total_documents/nq-open-randomized-uf-llama-predictions.jsonl.gz \
    --output-path ./qa_predictions/50_total_documents/nq-open-randomized-uf-llama-predictions-scored.jsonl.gz
```