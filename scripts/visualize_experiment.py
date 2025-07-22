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
        SELECT id, code, score, gen, parent_id, experiment_id, failure_type, retry_count, total_evaluation_time, generation_time, total_llm_time, total_tokens
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
    
    # Filter out failed programs (those with failure_type not None)
    successful_programs = [p for p in programs if p['failure_type'] is None]
    failed_programs = [p for p in programs if p['failure_type'] is not None]
    
    if not successful_programs:
        print("No successful programs found for this experiment.")
        return
    
    # Extract data from successful programs only
    generations = [p['gen'] for p in successful_programs]
    scores = [p['score'] for p in successful_programs]
    
    # Get all distinct failure types
    failure_types = set()
    for p in failed_programs:
        if p['failure_type'] is not None:
            failure_types.add(p['failure_type'])
    failure_types = sorted(list(failure_types))
    
    # Create figure with multiple subplots (3x2 layout for 6 graphs)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. Combined Score Analysis
    # Group scores by generation
    gen_to_scores = {}
    for gen, score in zip(generations, scores):
        if gen not in gen_to_scores:
            gen_to_scores[gen] = []
        gen_to_scores[gen].append(score)
    
    best_gens = sorted(gen_to_scores.keys())
    best_scores = [max(gen_to_scores[gen]) for gen in best_gens]
    avg_scores = [np.mean(gen_to_scores[gen]) for gen in best_gens]
    
    # Plot all scores as scatter
    ax1.scatter(generations, scores, alpha=0.4, color='blue', label='All Scores')
    
    # Plot best scores line
    ax1.plot(best_gens, best_scores, 'o-', color='red', label='Best Score')
    
    # Plot average scores line
    ax1.plot(best_gens, avg_scores, 'o-', color='green', label='Average Score')
    
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Score')
    ax1.set_title('Score Evolution Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Success and Failure Rates per Generation
    
    # Group all programs by generation
    gen_to_programs = {}
    for p in programs:
        gen = p['gen']
        if gen not in gen_to_programs:
            gen_to_programs[gen] = []
        gen_to_programs[gen].append(p)
    
    all_gens = sorted(gen_to_programs.keys())
    
    # Calculate success rate per generation
    success_rates = []
    for gen in all_gens:
        gen_programs = gen_to_programs[gen]
        successful_count = sum(1 for p in gen_programs if p['failure_type'] is None)
        success_rate = (successful_count / len(gen_programs)) * 100
        success_rates.append(success_rate)
    
    # Plot success rate
    ax2.plot(all_gens, success_rates, 'o-', linewidth=2, markersize=6, color='green', label='Success Rate')
    
    # Calculate and plot failure rates for each failure type
    colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, failure_type in enumerate(failure_types):
        failure_rates = []
        for gen in all_gens:
            gen_programs = gen_to_programs[gen]
            failure_count = sum(1 for p in gen_programs if p['failure_type'] == failure_type)
            failure_rate = (failure_count / len(gen_programs)) * 100
            failure_rates.append(failure_rate)
        
        color = colors[i % len(colors)]
        ax2.plot(all_gens, failure_rates, 'o-', linewidth=2, markersize=4, color=color, label=f'{failure_type}')
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Success and Failure Rates per Generation')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    # 3. Average retry count per generation
    retry_counts = [p['retry_count'] for p in programs]
    gen_to_retries = {}
    for gen, retries in zip(generations + [p['gen'] for p in failed_programs], retry_counts):
        if gen not in gen_to_retries:
            gen_to_retries[gen] = []
        gen_to_retries[gen].append(retries)
    
    all_retry_gens = sorted(gen_to_retries.keys())
    avg_retries = [np.mean(gen_to_retries[gen]) for gen in all_retry_gens]
    
    ax3.plot(all_retry_gens, avg_retries, 'o-', linewidth=2, markersize=6, color='purple')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Average Retry Count')
    ax3.set_title('Average Retry Count per Generation')
    ax3.grid(True, alpha=0.3)
    
    # 4. Average total tokens per generation
    total_tokens_list = [p['total_tokens'] for p in programs if p['total_tokens'] is not None]
    gen_to_tokens = {}
    for p in programs:
        if p['total_tokens'] is not None:
            gen = p['gen']
            if gen not in gen_to_tokens:
                gen_to_tokens[gen] = []
            gen_to_tokens[gen].append(p['total_tokens'])
    
    all_token_gens = sorted(gen_to_tokens.keys())
    avg_tokens = [np.mean(gen_to_tokens[gen]) / 1000 for gen in all_token_gens]  # Convert to thousands
    
    ax4.plot(all_token_gens, avg_tokens, 'o-', linewidth=2, markersize=6, color='green')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Average Total Tokens (thousands)')
    ax4.set_title('Average Total Tokens per Generation')
    ax4.grid(True, alpha=0.3)
    
    # 5. Time breakdown comparison (stacked bar chart)
    # Get all generations that have data for all time metrics
    generation_times = [p['generation_time'] for p in programs if p['generation_time'] is not None]
    total_evaluation_times = [p['total_evaluation_time'] for p in programs if p['total_evaluation_time'] is not None]
    total_llm_times = [p['total_llm_time'] for p in programs if p['total_llm_time'] is not None]
    
    # Group by generation
    gen_to_times = {}
    for p in programs:
        if p['generation_time'] is not None and p['total_evaluation_time'] is not None and p['total_llm_time'] is not None:
            gen = p['gen']
            if gen not in gen_to_times:
                gen_to_times[gen] = {'gen_times': [], 'eval_times': [], 'llm_times': []}
            gen_to_times[gen]['gen_times'].append(p['generation_time'])
            gen_to_times[gen]['eval_times'].append(p['total_evaluation_time'])
            gen_to_times[gen]['llm_times'].append(p['total_llm_time'])
    
    if gen_to_times:
        all_gens = sorted(gen_to_times.keys())
        avg_gen_times = [np.mean(gen_to_times[gen]['gen_times']) for gen in all_gens]
        avg_eval_times = [np.mean(gen_to_times[gen]['eval_times']) for gen in all_gens]
        avg_llm_times = [np.mean(gen_to_times[gen]['llm_times']) for gen in all_gens]
        
        # Calculate "other" time (generation time minus evaluation and LLM time)
        other_times = [gen_time - eval_time - llm_time for gen_time, eval_time, llm_time in zip(avg_gen_times, avg_eval_times, avg_llm_times)]
        
        x = range(len(all_gens))
        ax5.bar(x, avg_eval_times, label='Total Evaluation Time', color='orange', alpha=0.7)
        ax5.bar(x, avg_llm_times, bottom=avg_eval_times, label='Total LLM Time', color='purple', alpha=0.7)
        ax5.bar(x, other_times, bottom=[e+l for e, l in zip(avg_eval_times, avg_llm_times)], label='Other Time', color='gray', alpha=0.7)
        
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Time (s)')
        ax5.set_title('Generation Time Breakdown')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_xticks(x)
        ax5.set_xticklabels(all_gens)
    else:
        ax5.text(0.5, 0.5, 'No complete time data\navailable', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Generation Time Breakdown')
    
    # 6. Distribution of Total Generation Times
    generation_times = [p['generation_time'] for p in programs if p['generation_time'] is not None]
    
    if generation_times:
        ax6.hist(generation_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax6.axvline(np.mean(generation_times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(generation_times):.2f}s')
        ax6.axvline(np.median(generation_times), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(generation_times):.2f}s')
        ax6.set_xlabel('Generation Time (s)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Distribution of Total Generation Times')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No generation time data\navailable', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Distribution of Total Generation Times')
    
    # # Add statistics text
    # avg_retry_count = np.mean(retry_counts) if retry_counts else 0
    # avg_exec_time = np.mean(execution_times) if execution_times else 0
    # 
    # stats_text = f"""
    # Total Programs: {len(programs)}
    # Successful: {len(successful_programs)}
    # Failed: {len(failed_programs)}
    # Success Rate: {len(successful_programs)/len(programs)*100:.1f}%
    # Generations: {len(best_gens)}
    # Best Score: {max(scores):.4f}
    # Average Score: {np.mean(scores):.4f}
    # Score Std Dev: {np.std(scores):.4f}
    # Average Retry Count: {avg_retry_count:.1f}
    # Average Evaluation Time: {avg_exec_time:.2f}s
    # """
    # fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add vertical padding between subplots
    plt.subplots_adjust(hspace=0.6)
    
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
    
    # Filter programs for statistics
    successful_programs = [p for p in programs if p['failure_type'] is None]
    failed_programs = [p for p in programs if p['failure_type'] is not None]
    
    print(f"Found {len(programs)} total programs for experiment '{args.experiment}'")
    print(f"  - Successful: {len(successful_programs)}")
    print(f"  - Failed: {len(failed_programs)}")
    print(f"  - Success rate: {len(successful_programs)/len(programs)*100:.1f}%")
    
    if successful_programs:
        print(f"Generations: {min(p['gen'] for p in programs)} to {max(p['gen'] for p in programs)}")
        print(f"Score range: {min(p['score'] for p in successful_programs):.4f} to {max(p['score'] for p in successful_programs):.4f}")
    
    if failed_programs:
        # Count failure types
        failure_counts = {}
        for p in failed_programs:
            failure_type = p['failure_type']
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
        
        print("Failure breakdown:")
        for failure_type, count in sorted(failure_counts.items()):
            print(f"  - {failure_type}: {count}")
    
    # Create visualization
    create_visualization(programs, experiment['label'], args.output)


# Example usage:
# python scripts/visualize_experiment.py -e book-scanning --config examples/book_scanning/config.yml
if __name__ == "__main__":
    main() 