## Fibonacci Toy Problem

A starter task for AlphaEvolve-Lite.

* **solution.py** – contains a naïve recursive `fib` implementation wrapped by `# EVOLVE-START/END` markers.  
  The evolutionary loop rewrites only this block.

* **evaluate.py** – checks the first 20 Fibonacci numbers. A correct programme scores **1.0**, otherwise **0.0**.

Run the experiment with:

```bash
python scripts/run.py configs/fibonacci.yml
```