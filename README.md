# AlphaEvolveLite
An open source simplified implementation of the AlphaEvolve evolutionary coding agent.

## Directory Layout

```
AlphaEvolveLite/
├── alphaevolve/          # Core package
│   ├── **init**.py
│   ├── controller.py     # EvolutionController: orchestrates generations
│   ├── individual_generator.py # Individual generation pipeline (process-safe)
│   ├── db.py             # EvolutionaryDatabase: store candidates & evals
│   ├── prompts.py        # PromptSampler: builds LLM prompts from archive + user seed
│   ├── llm.py            # LLMEngine: wrapper for OpenAI/Ollama etc.
│   ├── patcher.py        # PatchApplier: applies diff blocks, syntax checks
│   ├── problem.py        # ProblemAPI: user-supplied evaluate() + block markers
│   ├── config.py         # Dataclass-backed config loader
│   └── log.py            # Opinionated logging setup
├── scripts/
│   ├── init\_db.py       # Creates tables & indices in PostgreSQL
│   └── run.py            # CLI entry-point: `python -m scripts.run config.yml`
├── examples/
│   ├── fibonacci/
│   │   ├── solution.py   # Starter code with EVOLVE markers
│   │   ├── evaluate.py   # Reference evaluator
│   │   └── config.yml    # Experiment config
│   └── README.md
├── tests/                # Pytest smoke tests for each module
├── pyproject.toml        # Poetry/PEP 518 build metadata
└── README.md             # (this file)
````

## Quick Start

1. **Install**

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
````

2. **Run PoC**

```bash
python -m scripts.run examples/fibonacci/config.yml
```

## Core Components

| Module                           | Responsibility                                                                                                    |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `db.EvolutionaryDatabase`        | CRUD for programs, metrics, and prompt cache. Minimal schema: `programs(id, code, score, gen, parent_id)`         |
| `prompts.PromptSampler`          | Pull top-k & random elites, build prompt with block context, return to `LLMEngine`.                               |
| `llm.LLMEngine`                  | Single `generate(prompt) → diff` using chosen provider.                                                           |
| `patcher.PatchApplier`           | Apply diff, run syntax lint; returns valid code or `None`.                                                        |
| `problem.ProblemAPI`             | User implements `evaluate(path) → float` and tags evolvable regions with `# EVOLVE-START/END`.                    |
| `controller.EvolutionController` | Loop: sample parent → prompt LLM → patch → eval → store → iterate. Simple Boltzmann selection; single population. |
| `individual_generator.*`         | Process-safe individual generation pipeline for parallel execution.                                               |
| `config.Config` & `log.*`        | YAML → dataclass; structured logging to console/file.                                                             |

## Configuration Snippet

```yaml
db_uri: postgresql://user:pass@localhost/alphaevolve
llm:
  provider: openai
  model: gpt-4o-mini
evolution:
  population_size: 20
  temperature: 1.5
  max_generations: 50
problem:
  entry_script: examples/fibonacci/solution.py
  evaluator:  examples/fibonacci/evaluate.py
```

## Contributing

Issues and pull requests are welcome. The PoC deliberately avoids parallelism; focus contributions on code clarity, better evaluators, and API polish.

## Licence

MIT
