# Simple Fibonacci module with an evolvable block.

def fib(n: int) -> int:
    # EVOLVE-START
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
    # EVOLVE-END


if __name__ == "__main__":
    for i in range(10):
        print(i, fib(i))
