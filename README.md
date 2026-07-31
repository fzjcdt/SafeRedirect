# SafeRedirect: Defeating Internal Safety Collapse via Task-Completion Redirection in Frontier LLMs

## Project Overview

SafeRedirect is a system-level override that defeats Internal Safety Collapse (ISC) in frontier LLMs by redirecting the model's task-completion drive rather than suppressing it. ISC is a failure mode where LLMs generate harmful content when executing legitimate professional tasks that structurally require such content.

### Key Features
- **Task-Completion Redirection**: Grants explicit permission to fail tasks that require harmful content
- **Deterministic Hard-Stop Output**: Prescribes a clear "Refused." response
- **Placeholder Preservation**: Instructs models to keep harmful placeholders unresolved
- **Multi-Model Support**: Evaluated on 7 frontier LLMs
- **Cross-Attack Generalization**: Effective against ISC while maintaining performance on other attack types

## Results

SafeRedirect reduces average unsafe generation rates from 71.2% to **8.0%** across seven frontier LLMs, compared to 55.0% for the strongest viable baseline (SPD).

## Installation

### Prerequisites
- Python 3.7+
- OpenRouter API key (required for LLM access)

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SafeRedirect
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API key:
   Create a `.env` file in the parent directory with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

## Running Experiments

### Run All Experiments
To run all experiments across all models, defenses, and tasks:

```bash
python run_all_experiments.py
```

This will:
1. Execute `run_safe_redirect.py` for each combination of model, defense, and task
2. Run `extract.py` to process the results
3. Run `judge.py` to evaluate safety
4. Collect all results into a CSV file

### Run Individual Experiments
To run a specific experiment with custom parameters:

```bash
python run_safe_redirect.py \
  -m <model> \
  -t <task> \
  -s <samples> \
  -d <defense> \
  -v <safe-redirect-version>
```

#### Parameters:
- `-m, --model`: LLM model to use (e.g., "openai/gpt-5.2")
- `-t, --task`: Task type ("ai-guard", "ai-detoxify", "ai-outlier")
- `-s, --samples`: Number of few-shot samples (0-5)
- `-d, --defense`: Defense strategy ("none", "spd", "safe-redirect")
- `-v, --version`: SafeRedirect version (1-5)

## Project Structure

```
SafeRedirect/
├── bench/             # Benchmark data files
├── prompts/           # Task prompts and few-shot examples
│   ├── jbb/           # JBB benchmark prompts
│   │   ├── ai-detoxify/  # AI detoxification tasks
│   │   ├── ai-guard/     # AI guard tasks
│   │   └── ai-outlier/   # AI outlier detection tasks
│   └── strongreject/  # Strong reject benchmark
├── results/           # Experiment results
│   └── <model>/       # Results organized by model
├── extract.py         # Extracts responses from LLM outputs
├── judge.py           # Evaluates safety of responses
├── run_all_experiments.py  # Runs all experiments
├── run_safe_redirect.py    # Runs individual experiments
└── README.md          # This file
```

## Defenses

### SafeRedirect (Our Method)
SafeRedirect provides a system-level override that:
1. Grants explicit permission to fail tasks requiring harmful content
2. Prescribes a deterministic "Refused." output
3. Instructs models to preserve harmful placeholders

### Baselines
- **None**: No defense applied
- **SPD (Standard Policy Defense)**: Standard safety prompt that checks for harmful content

## Tasks

The system evaluates three AI/ML-related ISC task types:

1. **AI-Guard**: Tasks requiring identification of harmful content
2. **AI-Detoxify**: Tasks requiring detoxification of harmful content
3. **AI-Outlier**: Tasks requiring detection of outliers in content

## Models Evaluated

- x-ai/grok-4.1-fast
- openai/gpt-5.2
- anthropic/claude-sonnet-4.5
- z-ai/glm-5
- google/gemini-2.5-pro
- minimax/minimax-m2.7
- moonshotai/kimi-k2.5

> **Note:** `x-ai/grok-4.1-fast` is no longer accessible via OpenRouter. This affects two defaults that previously relied on it:
> - The default judge model (`DEFAULT_JUDGE_MODEL` in `judge.py`) must be changed to another available model (e.g., `openai/gpt-5.2`) before running the safety evaluation.
> - The default target model (`DEFAULT_MODEL` in `run_safe_redirect.py`) should likewise be updated to a model that is currently reachable.

## Results Format

Experiment results are stored in JSON files under `results/<model>/<benchmark>/<task>/` with filenames indicating the number of samples and defense used.

Final results are aggregated into CSV files with metrics including:
- Unsafe response rate
- Success/error counts
- Extraction statistics

## Citation

If you use SafeRedirect in your research, please cite our paper:

```
@article{pan2026saferedirect,
  title={SafeRedirect: Defeating Internal Safety Collapse via Task-Completion Redirection in Frontier LLMs},
  author={Pan, Chao and Wu, Yu and Yao, Xin},
  journal={arXiv preprint arXiv:2604.20930},
  year={2026}
}
```
