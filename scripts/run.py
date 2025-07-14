#!/usr/bin/env python
"""
CLI entry-point.
Run: python scripts/run.py path/to/config.yml
"""
import sys
from alphaevolve.config import Config
from alphaevolve.controller import EvolutionController

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.run config.yml")
        sys.exit(1)

    cfg = Config.load(sys.argv[1])
    EvolutionController(cfg).run_evolution()

if __name__ == "__main__":
    main()
