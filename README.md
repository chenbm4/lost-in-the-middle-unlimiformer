# Lost in the Middle: How Language Models Use Long Contexts

This repository contains accompanying material for [Lost in the Middle: How
Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172).

## Table of Contents

- [Installation](#installation)
- [Multi-Document Question Answering Experiments](#multi-document-question-answering-experiments)
- [Multi-Document Question Answering Data](#multi-document-question-answering-data)
  * [Generating new multi-document QA data.](#generating-new-multi-document-qa-data)
- [References](#references)

## Installation

1. Set up a conda environment

``` sh
conda create -n lost-in-the-middle python=3.9 --yes
conda activate lost-in-the-middle
```

2. Install package and requirements

``` sh
pip install -e .
```

3. (optional) set up pre-commit hooks for development.

``` sh
pre-commit install
```

4. Change directory to Unlimiformer submodule and install requirements based on README.

cd unlimiformer

## Multi-Document Question Answering Experiments

See [EXPERIMENTS.md](./EXPERIMENTS.md#multi-document-question-answering) for
instructions to run and evaluate models on the multi-document QA task.

## Multi-Document Question Answering Data

[`qa_data/`](./qa_data/) contains multi-document question answering data for the
oracle setting (1 input document, which is exactly the passage the answers the
question) and 10-, 20-, and 30-document settings (where 1 input passage answers
the question, and the other passages do not contain an NQ-annotated answer).

Each line of this gzipped file is in the following format:

``` sh
{
  "question": "who got the first nobel prize in physics",
  "answers": [
    "Wilhelm Conrad Röntgen"
  ],
  "ctxs": [
    ...
    {
      "id": <string id, e.g., "71445">,
      "title": <string title of the wikipedia article that this passage comes from>,
      "text": <string content of the passage>,
      "score": <string relevance score, e.g. "1.0510446">,
      "hasanswer": <boolean, whether any of the values in the `answers` key appears in the text>,
      "original_retrieval_index": <int indicating the original retrieval index. for example, a value of 0 indicates that this was the top retrieved document>,
      "isgold": <boolean, true or false indicating if this chunk is the gold answer from NaturalQuestions>
    },
    ...
  ],
  "nq_annotated_gold": {
    "title": <string title of the wikipedia article containing the answer, as annotated in NaturalQuestions>,
    "long_answer": "<string content of the paragraph element containing the answer, as annotated in NaturalQuestions>",
    "chunked_long_answer": "<string content of the paragraph element containing the answer, randomly chunked to approximately 100 words>",
    "short_answers": [
      <string short answers, as annootated in NaturalQuestions>
    ]
  }
}
```

### Generating new multi-document QA data.

1. First, download Contriever retrieval results for each of the queries:

``` sh
wget https://nlp.stanford.edu/data/nfliu/lost-in-the-middle/nq-open-contriever-msmarco-retrieved-documents.jsonl.gz
```

2. Then, to generate examples with 20 total documents with the relevant documents at positions 0, 4, 9, 14, and 19, run:

``` sh
for gold_index in 0 4 9 14 19; do
    python -u ./scripts/make_qa_data_from_retrieval_results.py \
        --input-path nq-open-contriever-msmarco-retrieved-documents.jsonl.gz \
        --num-total-documents 20 \
        --gold-index ${gold_index} \
        --output-path qa_data/nq-open-20_total_documents_gold_at_${gold_index}.jsonl.gz
done
```

### Analysis

View ./scripts/experiment_analysis.ipynb for plotting and analysis of experiments.

Run

```
python ./scripts/count_average_tokens.py ./qa_prompts/open_oracle/simplified_prompts.jsonl.gz
```

on a ./qa_prompts/ file to get the average number of tokens per prompt.

## References

Here is a reference to the base paper and repository which this fork is based on.

```
@misc{liu-etal:2023:arxiv,
  author    = {Nelson F. Liu  and  Kevin Lin  and  John Hewitt  and Ashwin Paranjape  and Michele Bevilacqua  and  Fabio Petroni  and  Percy Liang},
  title     = {Lost in the Middle: How Language Models Use Long Contexts},
  note      = {arXiv:2307.03172},
  year      = {2023}
}
```
