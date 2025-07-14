# Experiment Visualization Script

This script provides comprehensive visualization of evolutionary algorithm results, showing how scores evolve across generations.

## Installation

First, install the required dependencies:

```bash
pip install matplotlib numpy
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Usage

### List all experiments
```bash
python scripts/visualize_experiment.py --list-experiments --config path/to/your/config.yml
```

### Visualize a specific experiment
```bash
python scripts/visualize_experiment.py --experiment "your_experiment_label"
```

### Save plot to file
```bash
python scripts/visualize_experiment.py --experiment "your_experiment_label" --output evolution_plot.png
```

### Use a different config file
```bash
python scripts/visualize_experiment.py --experiment "your_experiment_label" --config path/to/config.yml
```

## What the visualization shows

The script creates a 2x2 grid of plots:

1. **All Programs: Score vs Generation** - Scatter plot showing every program's score across generations
2. **Best Score per Generation** - Line plot showing the best score achieved in each generation
3. **Average Score per Generation** - Line plot showing the average score for each generation
4. **Score Distribution** - Histogram showing the distribution of all scores

Additional statistics are displayed in a text box:
- Total number of programs
- Number of generations
- Best score achieved
- Average score
- Standard deviation of scores

## Example

If you have an experiment called "fibonacci_optimization", you can visualize it with:

```bash
python scripts/visualize_experiment.py --experiment "fibonacci_optimization" --config examples/fibonacci/config.yml --output fibonacci_results.png
```

This will create a comprehensive visualization saved as `fibonacci_results.png` showing how your evolutionary algorithm performed over time. 