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


def group_data_by_generation(programs: list, key: str, filter_successful: bool = True):
    """Helper function to group data by generation and calculate statistics."""
    data_to_use = [p for p in programs if p['failure_type'] is None] if filter_successful else programs
    
    gen_to_values = {}
    for p in data_to_use:
        if p[key] is not None:  # Skip None values
            gen = p['gen']
            if gen not in gen_to_values:
                gen_to_values[gen] = []
            gen_to_values[gen].append(p[key])
    
    if not gen_to_values:
        return [], [], [], [], []
    
    gens = sorted(gen_to_values.keys())
    means = [np.mean(gen_to_values[gen]) for gen in gens]
    p10s = [np.percentile(gen_to_values[gen], 10) for gen in gens]
    p90s = [np.percentile(gen_to_values[gen], 90) for gen in gens]
    
    return gens, means, p10s, p90s, gen_to_values


def plot_with_percentiles(ax, gens, means, p10s, p90s, title, ylabel, color='blue', 
                         show_best=False, best_values=None, scale_factor=1.0):
    """Helper function to create a plot with percentile shading and summary lines."""
    # Apply scale factor if needed
    means = [m * scale_factor for m in means]
    p10s = [p * scale_factor for p in p10s]
    p90s = [p * scale_factor for p in p90s]
    
    # Plot percentile shading
    ax.fill_between(gens, p10s, p90s, alpha=0.3, color=color, label='10th-90th Percentile')
    
    # Plot mean line
    ax.plot(gens, means, '-', linewidth=2, color='red', label='Mean')
    
    # Plot percentile boundary lines (without individual labels)
    ax.plot(gens, p10s, '--', color=color, alpha=0.7)
    ax.plot(gens, p90s, '--', color=color, alpha=0.7)
    
    # Plot best line if requested
    if show_best and best_values:
        best_values = [b * scale_factor for b in best_values]
        ax.plot(gens, best_values, '-', linewidth=2, color='orange', label='Best')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_success_failure_rates(ax, programs, failure_types):
    """Helper function to plot success and failure rates per generation."""
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
    ax.plot(all_gens, success_rates, '-', linewidth=2, color='green', label='Success Rate')
    
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
        ax.plot(all_gens, failure_rates, '-', linewidth=2, color=color, label=f'{failure_type}')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Success and Failure Rates per Generation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 100)


def create_visualization(programs: list, experiment_label: str, output_path: str | None = None, 
                        show_individual: bool = True, show_combined: bool = True):
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
    fig = plt.figure(figsize=(15, 20))
    
    # Create a grid layout with extra space at the top for stats
    gs = fig.add_gridspec(4, 2, height_ratios=[0.15, 1, 1, 1], hspace=0.6)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[1, 0])  # First plot (top left)
    ax2 = fig.add_subplot(gs[1, 1])  # Second plot (top right)
    ax3 = fig.add_subplot(gs[2, 0])  # Third plot (middle left)
    ax4 = fig.add_subplot(gs[2, 1])  # Fourth plot (middle right)
    ax5 = fig.add_subplot(gs[3, 0])  # Fifth plot (bottom left)
    ax6 = fig.add_subplot(gs[3, 1])  # Sixth plot (bottom right)
    
    # 1. Score Analysis
    score_gens, score_means, score_p10s, score_p90s, gen_to_scores = group_data_by_generation(programs, 'score')
    if score_gens:
        best_scores = [max(gen_to_scores[gen]) for gen in score_gens]
        plot_with_percentiles(ax1, score_gens, score_means, score_p10s, score_p90s, 
                            'Score Evolution with Percentiles', 'Score', 
                            show_best=True, best_values=best_scores)
    
    # 2. Success and Failure Rates per Generation
    plot_success_failure_rates(ax2, programs, failure_types)
    
    # 3. Retry count per generation with percentiles
    retry_gens, retry_means, retry_p10s, retry_p90s, _ = group_data_by_generation(programs, 'retry_count', filter_successful=False)
    if retry_gens:
        plot_with_percentiles(ax3, retry_gens, retry_means, retry_p10s, retry_p90s, 
                            'Retry Count with Percentiles', 'Retry Count')
    
    # 4. Total tokens per generation with percentiles
    token_gens, token_means, token_p10s, token_p90s, _ = group_data_by_generation(programs, 'total_tokens', filter_successful=False)
    if token_gens:
        plot_with_percentiles(ax4, token_gens, token_means, token_p10s, token_p90s, 
                            'Total Tokens with Percentiles', 'Total Tokens (thousands)', 
                            scale_factor=1/1000)  # Convert to thousands
    
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
    
    # Add statistics text in dedicated space above plots
    # Calculate statistics using the data we already have
    total_tokens_list = [p['total_tokens'] for p in programs if p['total_tokens'] is not None]
    generation_times = [p['generation_time'] for p in programs if p['generation_time'] is not None]
    
    avg_total_tokens = np.mean(total_tokens_list) if total_tokens_list else 0
    avg_gen_time = np.mean(generation_times) if generation_times else 0
    
    stats_text = f"""
    Total Programs: {len(programs)}
    Best Score: {max(scores):.4f}
    Average Score: {np.mean(scores):.4f}
    Average Total Tokens: {avg_total_tokens:.2f}
    Average Generation Time: {avg_gen_time:.2f}s
    """
    
    # Create a dedicated subplot for stats at the top
    stats_ax = fig.add_subplot(gs[0, :])  # Span both columns
    stats_ax.axis('off')  # Hide axes
    stats_ax.text(0.02, 0.5, stats_text, fontsize=10, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 transform=stats_ax.transAxes)
    
    # Gridspec handles spacing automatically
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {output_path}")
    else:
        plt.show()
    
    # Create individual plots if requested
    if show_individual:
        create_individual_plots(programs, experiment_label, output_path)


def create_individual_plots(programs: list, experiment_label: str, output_path: str | None = None):
    """Create individual plots for each metric."""
    if not programs:
        return
    
    # Get data for each metric
    score_gens, score_means, score_p10s, score_p90s, gen_to_scores = group_data_by_generation(programs, 'score')
    retry_gens, retry_means, retry_p10s, retry_p90s, _ = group_data_by_generation(programs, 'retry_count', filter_successful=False)
    token_gens, token_means, token_p10s, token_p90s, _ = group_data_by_generation(programs, 'total_tokens', filter_successful=False)
    
    # Get failure types for success/failure rate plot
    failed_programs = [p for p in programs if p['failure_type'] is not None]
    failure_types = set()
    for p in failed_programs:
        if p['failure_type'] is not None:
            failure_types.add(p['failure_type'])
    failure_types = sorted(list(failure_types))
    
    # Get time data for time breakdown plot
    gen_to_times = {}
    for p in programs:
        if p['generation_time'] is not None and p['total_evaluation_time'] is not None and p['total_llm_time'] is not None:
            gen = p['gen']
            if gen not in gen_to_times:
                gen_to_times[gen] = {'gen_times': [], 'eval_times': [], 'llm_times': []}
            gen_to_times[gen]['gen_times'].append(p['generation_time'])
            gen_to_times[gen]['eval_times'].append(p['total_evaluation_time'])
            gen_to_times[gen]['llm_times'].append(p['total_llm_time'])
    
    # Create individual plots for all metrics
    plots_data = [
        (score_gens, score_means, score_p10s, score_p90s, 'Score Evolution', 'Score', True, 
         [max(gen_to_scores[gen]) for gen in score_gens] if score_gens else None, 1.0, 'score_evolution'),
        (retry_gens, retry_means, retry_p10s, retry_p90s, 'Retry Count', 'Retry Count', False, None, 1.0, 'retry_count'),
        (token_gens, token_means, token_p10s, token_p90s, 'Total Tokens', 'Total Tokens (thousands)', False, None, 1/1000, 'total_tokens'),
    ]
    
    # Create individual plots for percentile-based metrics
    for i, (gens, means, p10s, p90s, title, ylabel, show_best, best_values, scale_factor, plot_id) in enumerate(plots_data):
        if not gens:
            continue
            
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_with_percentiles(ax, gens, means, p10s, p90s, f"{title} - {experiment_label}", ylabel, 
                            show_best=show_best, best_values=best_values, scale_factor=scale_factor)
        
        plt.tight_layout()
        
        if output_path:
            # Create individual output path
            base_path = Path(output_path)
            individual_path = base_path.parent / f"{base_path.stem}_{plot_id}{base_path.suffix}"
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            print(f"Individual plot saved to: {individual_path}")
        else:
            plt.show()
        
        plt.close()
    
    # Create individual success/failure rate plot
    if failure_types:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_success_failure_rates(ax, programs, failure_types)
        ax.set_title(f"Success and Failure Rates - {experiment_label}")
        
        plt.tight_layout()
        
        if output_path:
            base_path = Path(output_path)
            individual_path = base_path.parent / f"{base_path.stem}_success_failure_rates{base_path.suffix}"
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            print(f"Individual plot saved to: {individual_path}")
        else:
            plt.show()
        
        plt.close()
    
    # Create individual time breakdown plot
    if gen_to_times:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_gens = sorted(gen_to_times.keys())
        avg_gen_times = [np.mean(gen_to_times[gen]['gen_times']) for gen in all_gens]
        avg_eval_times = [np.mean(gen_to_times[gen]['eval_times']) for gen in all_gens]
        avg_llm_times = [np.mean(gen_to_times[gen]['llm_times']) for gen in all_gens]
        
        # Calculate "other" time (generation time minus evaluation and LLM time)
        other_times = [gen_time - eval_time - llm_time for gen_time, eval_time, llm_time in zip(avg_gen_times, avg_eval_times, avg_llm_times)]
        
        x = range(len(all_gens))
        ax.bar(x, avg_eval_times, label='Total Evaluation Time', color='orange', alpha=0.7)
        ax.bar(x, avg_llm_times, bottom=avg_eval_times, label='Total LLM Time', color='purple', alpha=0.7)
        ax.bar(x, other_times, bottom=[e+l for e, l in zip(avg_eval_times, avg_llm_times)], label='Other Time', color='gray', alpha=0.7)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'Generation Time Breakdown - {experiment_label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(all_gens)
        
        plt.tight_layout()
        
        if output_path:
            base_path = Path(output_path)
            individual_path = base_path.parent / f"{base_path.stem}_time_breakdown{base_path.suffix}"
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            print(f"Individual plot saved to: {individual_path}")
        else:
            plt.show()
        
        plt.close()
    
    # Create individual generation time distribution plot
    generation_times = [p['generation_time'] for p in programs if p['generation_time'] is not None]
    
    if generation_times:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(generation_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(generation_times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(generation_times):.2f}s')
        ax.axvline(np.median(generation_times), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(generation_times):.2f}s')
        ax.set_xlabel('Generation Time (s)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of Total Generation Times - {experiment_label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            base_path = Path(output_path)
            individual_path = base_path.parent / f"{base_path.stem}_generation_time_distribution{base_path.suffix}"
            plt.savefig(individual_path, dpi=300, bbox_inches='tight')
            print(f"Individual plot saved to: {individual_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--experiment', '-e', type=str, help='Experiment label to visualize')
    parser.add_argument('--list-experiments', '-l', action='store_true', help='List all available experiments')
    parser.add_argument('--output', '-o', type=str, help='Output file path for the plot (e.g., plot.png)')
    parser.add_argument('--config', '-c', type=str, default='config.yml', help='Config file path')
    parser.add_argument('--individual-only', action='store_true', help='Show only individual plots, not combined')
    parser.add_argument('--combined-only', action='store_true', help='Show only combined plot, not individual')
    
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
    
    # Determine what to show
    show_individual = not args.combined_only
    show_combined = not args.individual_only
    
    # Create visualization
    create_visualization(programs, experiment['label'], args.output, show_individual, show_combined)


# Example usage:
# python scripts/visualize_experiment.py -e book-scanning --config examples/book_scanning/config.yml
if __name__ == "__main__":
    main() 