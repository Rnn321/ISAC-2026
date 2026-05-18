# ISAC

This repository contains the data, scripts, baselines, and evaluation results for **ISAC**, a method proposed in our paper for commit message generation with issue-aware context. The repository is organized to support data collection, model generation under different issue settings, and automatic evaluation against reference commit messages.

## Repository Structure

```text
.
|-- data/
|   |-- experiment.json
|   |   # Experimental samples used for model inference and evaluation.
|   |-- ApacheCM-Issue/
|       |-- full_issues.json
|           # Full commit-issue dataset. Each item can be identified by commit_sha.
|-- issue/
|   |-- issue_crawler.py
|   |   # Crawls issue information from GitHub/Jira for Apache commits.
|   |-- deepseek.py
|       # Runs commit message generation experiments with DeepSeek/OpenAI  APIs.
|-- baselines/
|   |-- *_without_issue_363.jsonl
|       # Baseline outputs generated without issue information.
|-- results/
|   |-- with_full_issue_363_collection/
|   |   |-- *_with_full_issue_363.json
|   |   |-- metrics_363.csv
|   |   |-- metrics_363.json
|   |   |-- sbert_363_cache.json
|   |-- without_issue_363_collection/
|       |-- *_without_issue_363.json
|       |-- metrics_363.csv
|       |-- metrics_363.json
|       |-- sbert_363_cache.json
|-- utils_eval/
    |-- eval_baselines.py
    |   # Main evaluation entry point.
    |-- metric/
    |   # CIDEr and related metric implementations.
    |-- sbert/
        # SBERT-based semantic similarity calculation.
```

## Data

- `data/experiment.json` contains the experimental samples used in the paper experiments.
- `data/ApacheCM-Issue/full_issues.json` contains the full commit-issue data. It includes **76,484** records.

Each data item is organized around a commit and may include fields such as repository information, commit SHA, original commit message, issue reference, issue title, and issue body.

## Experiments

The generation script in `issue/deepseek.py` supports two main settings:

- `without_issue`: generate a commit message using only the code diff.
- `with_full_issue`: generate a commit message using the code diff together with the related issue title and issue body.

Example:

```bash
python issue/deepseek.py --provider deepseek --model deepseek-chat --experiment with_full_issue --input-file data/experiment.json --output-file results/example.json --max-workers 1
```

Before running model inference, configure the required API key, such as `DEEPSEEK_API_KEY` or `OPENAI_API_KEY`.

## Issue Collection

`issue/issue_crawler.py` collects issue information from GitHub and Jira for Apache project commits. It supports crawling from raw commit files and from retrieved similar-commit results.

Example:

```bash
python issue/issue_crawler.py --mode commits --input-file <input.jsonl> --output-file <output.json> --max-workers 1
```

## Evaluation

`utils_eval/eval_baselines.py` evaluates generated commit messages with automatic metrics, including BLEU, ROUGE-L, METEOR, CIDEr, and SBERT cosine similarity.

Example:

```bash
python utils_eval/eval_baselines.py <result.json>
```

## Results

The `results/` directory stores model outputs and metric summaries for the 363-sample evaluation set:

- `with_full_issue_363_collection/`: results generated with issue context.
- `without_issue_363_collection/`: results generated without issue context.

The `baselines/` directory stores baseline outputs for comparison.

## Notes

- Runtime artifacts such as `__pycache__/`, logs, and temporary caches are not part of the core data organization.
- Some scripts require external API access and should be configured with local credentials before use.
