#!/usr/bin/env python3
"""
Script to view LLM conversation history for a specific program.

Usage:
    python scripts/view_conversation.py --program-id 123
    python scripts/view_conversation.py --program-id 123 --pretty
    python scripts/view_conversation.py --list-programs --experiment "experiment_label"
"""

import argparse
import sqlite3
import json
import sys
from pathlib import Path

# Add the parent directory to the path so we can import alphaevolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphaevolve.config import Config


def get_programs(db_path: str, experiment_id: int = None) -> list:
    """Get list of programs from the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    if experiment_id:
        cursor.execute("""
            SELECT id, explanation, code, score, gen, parent_id, experiment_id, failure_type, retry_count, 
                   total_evaluation_time, generation_time, total_llm_time, total_tokens, conversation
            FROM programs
            WHERE experiment_id = ?
            ORDER BY gen DESC, score DESC
        """, (experiment_id,))
    else:
        cursor.execute("""
            SELECT id, explanation, code, score, gen, parent_id, experiment_id, failure_type, retry_count, 
                   total_evaluation_time, generation_time, total_llm_time, total_tokens, conversation
            FROM programs
            ORDER BY gen DESC, score DESC
        """)
    
    programs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return programs


def get_program_by_id(db_path: str, program_id: int) -> dict:
    """Get a specific program by ID."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, explanation, code, score, gen, parent_id, experiment_id, failure_type, retry_count, 
               total_evaluation_time, generation_time, total_llm_time, total_tokens, conversation
        FROM programs
        WHERE id = ?
    """, (program_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return None
    return dict(row)


def get_experiment_id(db_path: str, experiment_label: str) -> int:
    """Get experiment ID by label."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM experiments WHERE label = ?", (experiment_label,))
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return None
    return row['id']


def display_conversation(conversation_json: str, pretty: bool = False):
    """Display the conversation in a readable format."""
    if not conversation_json:
        print("No conversation data available for this program.")
        return
    
    try:
        conversation = json.loads(conversation_json)
        
        if pretty:
            print("=== LLM Conversation History ===\n")
            for i, message in enumerate(conversation):
                role = message.get('role', 'unknown').upper()
                content = message.get('content', '')
                
                print(f"[{i+1}] {role}:")
                print(f"{'=' * (len(role) + 3)}")
                print(content)
                print()
        else:
            print(json.dumps(conversation, indent=2))
            
    except json.JSONDecodeError as e:
        print(f"Error parsing conversation JSON: {e}")
        print("Raw conversation data:")
        print(conversation_json)


def main():
    parser = argparse.ArgumentParser(description="View LLM conversation history for programs")
    parser.add_argument("--db", default="alphaevolve.db", help="Database file path")
    parser.add_argument("--program-id", type=int, help="Program ID to view conversation for")
    parser.add_argument("--list-programs", action="store_true", help="List all programs")
    parser.add_argument("--experiment", help="Experiment label to filter by")
    parser.add_argument("--pretty", action="store_true", help="Pretty print conversation")
    
    args = parser.parse_args()
    
    if args.list_programs:
        experiment_id = None
        if args.experiment:
            experiment_id = get_experiment_id(args.db, args.experiment)
            if experiment_id is None:
                print(f"Experiment '{args.experiment}' not found.")
                return
        
        programs = get_programs(args.db, experiment_id)
        
        print("Programs in database:")
        print(f"{'ID':<5} {'Gen':<4} {'Score':<8} {'Has Conv':<8} {'Failure':<12}")
        print("-" * 50)
        
        for prog in programs:
            has_conv = "Yes" if prog['conversation'] else "No"
            score = f"{prog['score']:.3f}" if prog['score'] is not None else "N/A"
            failure = prog['failure_type'] or "Success"
            print(f"{prog['id']:<5} {prog['gen']:<4} {score:<8} {has_conv:<8} {failure:<12}")
    
    elif args.program_id:
        program = get_program_by_id(args.db, args.program_id)
        
        if program is None:
            print(f"Program with ID {args.program_id} not found.")
            return
        
        print(f"Program ID: {program['id']}")
        print(f"Explanation: {program['explanation']}")
        print(f"Generation: {program['gen']}")
        print(f"Score: {program['score']}")
        print(f"Parent ID: {program['parent_id']}")
        print(f"Failure Type: {program['failure_type']}")
        print(f"Retry Count: {program['retry_count']}")
        print(f"Generation Time: {program['generation_time']:.2f}s")
        print(f"Total LLM Time: {program['total_llm_time']:.2f}s")
        print(f"Total Tokens: {program['total_tokens']}")
        print()
        
        display_conversation(program['conversation'], args.pretty)    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 