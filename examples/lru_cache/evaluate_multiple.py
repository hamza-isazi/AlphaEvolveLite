#!/usr/bin/env python3
"""
Script to evaluate a program over multiple iterations and calculate average scores.
Usage: python evaluate_multiple.py <program_path> <num_iterations>
"""

import sys
import statistics
from evaluate import evaluate


def evaluate_multiple(program_path: str, num_iterations: int) -> dict:
    """
    Evaluate a program over multiple iterations and return average scores.
    
    Args:
        program_path: Path to the program to evaluate
        num_iterations: Number of iterations to run
        
    Returns:
        Dictionary with average scores for each metric
    """
    results = []
    
    print(f"Evaluating {program_path} over {num_iterations} iterations...")
    
    for i in range(num_iterations):
        try:
            result = evaluate(program_path)
            results.append(result)
            # print(f"Iteration {i+1}/{num_iterations}: {result}")
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}")
            continue
    
    if not results:
        raise ValueError("No successful evaluations completed")
    
    # Calculate averages for each metric
    avg_results = {}
    for metric in results[0].keys():
        values = [result[metric] for result in results]
        avg_results[f"avg_{metric}"] = statistics.mean(values)
        avg_results[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        avg_results[f"min_{metric}"] = min(values)
        avg_results[f"max_{metric}"] = max(values)
    
    # Add summary statistics
    avg_results["total_iterations"] = len(results)
    avg_results["successful_iterations"] = len(results)
    
    return avg_results


def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate_multiple.py <program_path> <num_iterations>")
        print("Example: python evaluate_multiple.py initial_program.py 10")
        sys.exit(1)
    
    program_path = sys.argv[1]
    try:
        num_iterations = int(sys.argv[2])
        if num_iterations <= 0:
            raise ValueError("Number of iterations must be positive")
    except ValueError as e:
        print(f"Error: Invalid number of iterations: {e}")
        sys.exit(1)
    
    try:
        results = evaluate_multiple(program_path, num_iterations)
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Program: {program_path}")
        print(f"Successful iterations: {results['successful_iterations']}/{results['total_iterations']}")
        print()
        
        # Print averages for each metric
        metrics = ['latency_sec', 'peak_memory_mb', 'score']
        for metric in metrics:
            print(f"{metric.upper()}:")
            print(f"  Average: {results[f'avg_{metric}']:.6f}")
            print(f"  Std Dev:  {results[f'std_{metric}']:.6f}")
            print(f"  Min:      {results[f'min_{metric}']:.6f}")
            print(f"  Max:      {results[f'max_{metric}']:.6f}")
            print()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 