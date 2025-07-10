# EVOLVE-BLOCK-START
# initial_program.py
import sys

def readints():
    return list(map(int, sys.stdin.readline().split()))

B, L, D = readints()
scores = readints()
libraries = []
for _ in range(L):
    N, T, M = readints()
    books = readints()
    libraries.append((T, M, books))

# Naive strategy: sign up all libraries in input order, scan all books
print(L)
for i, (T, M, books) in enumerate(libraries):
    books = books[:min(len(books), (D - T) * M)]
    print(f"{i} {len(books)}")
    print(" ".join(map(str, books)))
# EVOLVE-BLOCK-END
