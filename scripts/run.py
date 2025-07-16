#!/usr/bin/env python
"""
CLI entry-point.
Run: python scripts/run.py path/to/config.yml [--debug]
"""
import sys
import argparse
from alphaevolve.config import Config
from alphaevolve.controller import EvolutionController

def main():
    parser = argparse.ArgumentParser(description='Run AlphaEvolve evolution experiment')
    parser.add_argument('config', help='Path to configuration YAML file')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug logging (verbose individual-level output)')
    
    args = parser.parse_args()
    
    # Load config
    cfg = Config.load(args.config)
    
    # Override debug setting from command line
    cfg.debug = args.debug
    
    # Run evolution
    EvolutionController(cfg).run_evolution()

if __name__ == "__main__":
    main()
