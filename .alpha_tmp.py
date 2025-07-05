# Simple Fibonacci module with an evolvable block.

def fib(n: int, memo={}) -> int:
    # EVOLVE-START
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
    return memo[n]
    # EVOLVE-END


if __name__ == "__main__":
    for i in range(10):
        print(i, fib(i))
