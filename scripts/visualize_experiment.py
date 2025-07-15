#!/usr/bin/env python3
"""
Script to visualize score vs generation for a given experiment.

Usage:
    python scripts/visualize_experiment.py --experiment "experiment_label"
    python scripts/visualize_experiment.py --experiment "experiment_label" --output plot.png
    python scripts/visualize_experiment.py --list-experiments
"""

import argparse
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import alphaevolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphaevolve.config import Config


def get_experiments(db_path: str) -> list:
    """Get list of all experiments in the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, label, notes, started_at,
               (SELECT COUNT(*) FROM programs WHERE experiment_id = experiments.id) as program_count
        FROM experiments
        ORDER BY started_at DESC
    """)
    
    experiments = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return experiments


def get_experiment_data(db_path: str, experiment_id: int) -> list:
    """Get all programs for a specific experiment."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, code, score, gen, parent_id, experiment_id
        FROM programs
        WHERE experiment_id = ?
        ORDER BY gen, score DESC
    """, (experiment_id,))
    
    programs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return programs


def create_visualization(programs: list, experiment_label: str, output_path: str | None = None):
    """Create and save the visualization."""
    if not programs:
        print("No programs found for this experiment.")
        return
    
    # Extract data
    generations = [p['gen'] for p in programs]
    scores = [p['score'] for p in programs]
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Evolution Progress: {experiment_label}', fontsize=16, fontweight='bold')
    
    # 1. Score vs Generation (all points)
    ax1.scatter(generations, scores, alpha=0.6, s=20, color='blue')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Score')
    ax1.set_title('All Programs: Score vs Generation')
    ax1.grid(True, alpha=0.3)
    
    # 2. Best score per generation
    gen_to_scores = {}
    for gen, score in zip(generations, scores):
        if gen not in gen_to_scores:
            gen_to_scores[gen] = []
        gen_to_scores[gen].append(score)
    
    best_gens = sorted(gen_to_scores.keys())
    best_scores = [max(gen_to_scores[gen]) for gen in best_gens]
    
    ax2.plot(best_gens, best_scores, 'o-', linewidth=2, markersize=6, color='red')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Best Score')
    ax2.set_title('Best Score per Generation')
    ax2.grid(True, alpha=0.3)
    
    # 3. Average score per generation
    avg_scores = [np.mean(gen_to_scores[gen]) for gen in best_gens]
    ax3.plot(best_gens, avg_scores, 'o-', linewidth=2, markersize=6, color='green')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Average Score')
    ax3.set_title('Average Score per Generation')
    ax3.grid(True, alpha=0.3)
    
    # 4. Score distribution histogram
    ax4.hist(scores, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_xlabel('Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Score Distribution')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""
    Total Programs: {len(programs)}
    Generations: {len(best_gens)}
    Best Score: {max(scores):.4f}
    Average Score: {np.mean(scores):.4f}
    Score Std Dev: {np.std(scores):.4f}
    """
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout to prevent overlapping labels
    plt.tight_layout(rect=(0, 0.08, 1, 0.92), h_pad=3.0, w_pad=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--experiment', '-e', type=str, help='Experiment label to visualize')
    parser.add_argument('--list-experiments', '-l', action='store_true', help='List all available experiments')
    parser.add_argument('--output', '-o', type=str, help='Output file path for the plot (e.g., plot.png)')
    parser.add_argument('--config', '-c', type=str, default='config.yml', help='Config file path')
    
    args = parser.parse_args()
    
    # Load config to get database path
    try:
        config = Config.load(args.config)
        db_path = config.db_uri.replace("sqlite:///", "", 1)
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Please make sure you're in the correct directory with a config.yml file")
        return
    
    if args.list_experiments:
        print("Available experiments:")
        print("-" * 80)
        experiments = get_experiments(db_path)
        for exp in experiments:
            print(f"ID: {exp['id']}")
            print(f"Label: {exp['label']}")
            print(f"Notes: {exp['notes']}")
            print(f"Started: {exp['started_at']}")
            print(f"Programs: {exp['program_count']}")
            print("-" * 80)
        return
    
    if not args.experiment:
        print("Please specify an experiment label with --experiment or use --list-experiments to see available experiments")
        return
    
    # Find experiment by label
    experiments = get_experiments(db_path)
    experiment = None
    for exp in experiments:
        if exp['label'] == args.experiment:
            experiment = exp
            break
    
    if not experiment:
        print(f"Experiment '{args.experiment}' not found.")
        print("Available experiments:")
        for exp in experiments:
            print(f"  - {exp['label']}")
        return
    
    # Get experiment data
    programs = get_experiment_data(db_path, experiment['id'])
    
    if not programs:
        print(f"No programs found for experiment '{args.experiment}'")
        return
    
    print(f"Found {len(programs)} programs for experiment '{args.experiment}'")
    print(f"Generations: {min(p['gen'] for p in programs)} to {max(p['gen'] for p in programs)}")
    print(f"Score range: {min(p['score'] for p in programs):.4f} to {max(p['score'] for p in programs):.4f}")
    
    # Create visualization
    create_visualization(programs, experiment['label'], args.output)


if __name__ == "__main__":
    main() 