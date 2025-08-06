# AlphaEvolveLite
An open source simplified implementation of the AlphaEvolve evolutionary coding agent.

## Features

- **Evolutionary Program Generation**: Uses LLM-based mutation and selection to improve code
- **Tabu Search Integration**: Combines improvement and fundamentally different approaches to escape local optima
- **Full Conversation Storage**: Stores complete LLM conversation history for each generated program
- **Multiple LLM Providers**: Support for OpenAI and Google Gemini APIs
- **Flexible Evaluation**: Custom evaluation functions for any programming problem
- **Progress Tracking**: Detailed logging and visualization of evolution progress
- **Experiment Visualization**: Tools to analyze and visualize evolution results

## Directory Layout

```
AlphaEvolveLite/
├── alphaevolve/          # Core package
│   ├── __init__.py
│   ├── controller.py     # EvolutionController: orchestrates generations
│   ├── program_generator.py # Program generation pipeline (process-safe)
│   ├── db.py             # EvolutionaryDatabase: store candidates & evals
│   ├── prompts.py        # PromptSampler: builds LLM prompts from archive + user seed
│   ├── llm.py            # LLMEngine: wrapper for OpenAI/Gemini etc.
│   ├── patcher.py        # PatchApplier: applies diff blocks, syntax checks
│   ├── problem.py        # ProblemAPI: user-supplied evaluate() + block markers
│   ├── response_parser.py # ResponseParser: extracts code blocks from LLM responses
│   ├── config.py         # Dataclass-backed config loader
│   └── log.py            # Opinionated logging setup
├── scripts/
│   ├── run.py            # CLI entry-point: `python scripts/run.py config.yml [--debug]`
│   ├── view_conversation.py # View conversation history for programs
│   ├── visualize_experiment.py # Generate evolution visualization plots
│   └── README_visualization.md # Visualization documentation
├── examples/
│   ├── fibonacci/        # Simple Fibonacci sequence example
│   │   ├── solution.py   # Starter code with EVOLVE markers
│   │   ├── evaluate.py   # Reference evaluator
│   │   └── config.yml    # Experiment config
│   ├── book_scanning/    # HashCode 2020 optimization problem
│   │   ├── initial_program.py # Starter solution
│   │   ├── evaluate.py   # Evaluation script
│   │   ├── config.yml    # Experiment config
│   │   └── inputs/       # Test input files
│   └── timetable_optimization/ # Class scheduling optimization
│       ├── solution.py   # Starter solution
│       ├── evaluate.py   # Evaluation script
│       ├── config.yml    # Experiment config
│       └── inputs/       # Test input files
├── tests/                # Pytest smoke tests for each module
├── results/              # Generated experiment results and visualizations
├── pyproject.toml        # Setuptools build metadata
├── requirements.txt      # Python dependencies
└── README.md             # (this file)
```

## Quick Start

1. **Install**

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

2. **Run Example**

```bash
# Normal mode with progress bars and generation summaries
python scripts/run.py examples/fibonacci/config.yml

# Debug mode with verbose individual-level logging
python scripts/run.py examples/fibonacci/config.yml --debug
```

3. **View Conversation History**

```bash
# List all programs with conversation data
python scripts/view_conversation.py --list-programs

# View conversation for a specific program
python scripts/view_conversation.py --program-id 123 --pretty

# View conversation for programs in a specific experiment
python scripts/view_conversation.py --list-programs --experiment "fib-baseline-v1"
```

4. **Visualize Results**

```bash
# List all available experiments
python scripts/visualize_experiment.py --db alphaevolve.db --list-experiments

# Generate evolution visualization plots
python scripts/visualize_experiment.py --db alphaevolve.db --experiment "fib-baseline-v1"

# Save plots to file
python scripts/visualize_experiment.py --db alphaevolve.db --experiment "fib-baseline-v1" --output results/experiment_plot.png

# Limit to first 10 generations
python scripts/visualize_experiment.py --db alphaevolve.db --experiment "fib-baseline-v1" --max-generations 10

# Show only individual plots (not combined)
python scripts/visualize_experiment.py --db alphaevolve.db --experiment "fib-baseline-v1" --individual-only

# Show only combined plot (not individual)
python scripts/visualize_experiment.py --db alphaevolve.db --experiment "fib-baseline-v1" --combined-only
```

## Logging Features

The system provides two logging modes:

### Normal Mode (Default)
- **Progress bars**: TQDM progress bars for each generation
- **Generation summaries**: At the end of each generation, displays:
  - Success rate (successful/total individuals)
  - Average and best fitness scores
  - Failure breakdown (runtime errors, timeouts, patch failures, etc.)
- **Clean output**: Minimal clutter for large populations

### Debug Mode (`--debug` flag)
- **Verbose logging**: Individual-level details for each attempt
- **Retry information**: Patch and evaluation retry attempts
- **Error details**: Specific failure reasons and error messages
- **Full trace**: Complete evolution process visibility

## Core Components

| Module                           | Responsibility                                                                                                    |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `db.EvolutionaryDatabase`        | CRUD for programs, metrics, and prompt cache. Schema: `programs(id, code, score, gen, parent_id, conversation)`   |
| `prompts.PromptSampler`          | Pull top-k & random elites, build prompt with block context, return to `LLMEngine`.                               |
| `llm.LLMEngine`                  | Single `generate(prompt) → diff` using chosen provider. Maintains conversation history.                           |
| `patcher.PatchApplier`           | Apply diff, run syntax lint; returns valid code or `None`.                                                        |
| `response_parser.ResponseParser` | Extract code blocks and patches from LLM responses.                                                               |
| `problem.ProblemAPI`             | User implements `evaluate(path) → float` and tags evolvable regions with `# EVOLVE-START/END`.                    |
| `controller.EvolutionController` | Loop: sample parent → prompt LLM → patch → eval → store → iterate. Simple Boltzmann selection; single population. |
| `program_generator.*`            | Process-safe program generation pipeline for parallel execution.                                                   |
| `config.Config` & `log.*`        | YAML → dataclass; structured logging to console/file.                                                             |

## LLM Providers

AlphaEvolveLite supports multiple LLM providers and models through the `provider` field in the configuration. You can configure multiple models with selection probabilities for each generation.

### Supported Providers

#### OpenAI
**Environment Variable**: `OPENAI_API_KEY`
- `gpt-4o-mini`
- `gpt-4o`
- `gpt-4`
- `gpt-3.5-turbo`

#### Google Gemini
**Environment Variable**: `GOOGLE_API_KEY`
- `gemini-2.5-flash`
- `gemini-2.5-pro`
- `gemini-1.5-flash`
- `gemini-1.5-pro`

The Gemini integration uses the OpenAI-compatible API format provided by Google, making it seamless to switch between providers.

### Model Selection
For each generation, the system randomly selects a model based on the configured probabilities. This allows you to:
- Use faster, cheaper models for most generations
- Occasionally use more powerful models for complex problems
- Balance cost and performance based on your needs

### Retry Model Configuration
You can specify a dedicated model for retries and feedback using the `retry_model` parameter:
- **Retries**: When initial program generation fails, the system uses the retry model to generate improved versions
- **Feedback**: For successful programs, the retry model generates feedback to help guide future evolution
- **Fallback**: If no retry model is specified, the system uses the normal model selection logic

This is useful for:
- Using more capable models for error correction and feedback
- Reducing costs by using cheaper models for initial generation
- Ensuring high-quality feedback from the most capable available model

### Tabu Search Configuration
AlphaEvolveLite implements a tabu search inspired approach to program generation that helps escape local optima:

- **Improvement Mode** (default): The LLM tries to improve upon existing programs by combining the best ideas and adding novel improvements
- **Tabu Search Mode**: The LLM is instructed to take a fundamentally different approach, treating prior programs as "taboo" and exploring alternative algorithms, data structures, or problem-solving paradigms

The `tabu_search_probability` parameter controls the probability of using tabu search mode vs improvement mode for each program generation:
- `0.0`: Always use improvement mode (traditional evolution)
- `0.5`: 50% chance of each mode (balanced exploration/exploitation)
- `1.0`: Always use tabu search mode (maximum exploration)

This feature is particularly useful for:
- Escaping local optima when evolution gets stuck
- Exploring diverse solution approaches
- Balancing between incremental improvements and radical innovations

## Examples

### Fibonacci Sequence
A simple example demonstrating basic evolution concepts:
```bash
python scripts/run.py examples/fibonacci/config.yml
```

### Book Scanning Optimization
A complex optimization problem from HashCode 2020:
```bash
python scripts/run.py examples/book_scanning/config.yml
```

### Timetable Optimization
A class scheduling optimization problem:
```bash
python scripts/run.py examples/timetable_optimization/config.yml
```

## Configuration Snippet

```yaml
db_uri: sqlite:///alphaevolve.db

experiment:
  label: my-experiment-v1
  notes: "Description of the experiment"
  save_top_k: 5

llm:
  provider: openai
  llm_timeout: 120.0  # Global timeout for all LLM requests
  models:
    - name: gpt-4o-mini
      probability: 0.7
      temperature: 0.9
    - name: gpt-4o
      probability: 0.3
      temperature: 0.8
  retry_model: gpt-4o  # Use more capable model for retries and feedback
  system_prompt: |
    You are an expert software engineer solving the following challenge:
    [Your problem description here]

evolution:
  population_size: 40
  temperature: 1.2
  max_generations: 5
  inspiration_count: 3
  max_retries: 3
  eval_timeout: 60.0
  tabu_search_probability: 0.5  # Probability of using tabu search vs improvement approach

problem:
  entry_script: path/to/your/solution.py
  evaluator: path/to/your/evaluate.py
```

## License

MIT
