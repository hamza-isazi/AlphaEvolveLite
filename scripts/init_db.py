#!/usr/bin/env python
"""
Initialises PostgreSQL schema for AlphaEvolve-Lite.
Usage: python scripts/init_db.py postgresql://user:pass@host/db
"""
import sys
from alphaevolve.db import EvolutionaryDatabase
from alphaevolve.log import init_logger

logger = init_logger("init_db")

def main():
    if len(sys.argv) != 2:
        print("Provide the DB URI, e.g. postgresql://user:pass@localhost/alphaevolve")
        sys.exit(1)
    EvolutionaryDatabase(sys.argv[1])  # constructor creates schema
    logger.info("Schema ready")

if __name__ == "__main__":
    main()
