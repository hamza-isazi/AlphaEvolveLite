# Simple Fibonacci module with an evolvable block.

# EVOLVE-BLOCK-START
def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    for i in range(10):
        print(i, fib(i))
