import sys
from io import StringIO

def run(input_text: str) -> str:
    # Create a StringIO object to capture output
    output_buffer = StringIO()
    
    # Redirect stdout to our buffer
    original_stdout = sys.stdout
    sys.stdout = output_buffer
    
    try:
        # Create a StringIO object for input
        input_buffer = StringIO(input_text)
        
        def readints():
            return list(map(int, input_buffer.readline().split()))

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
            
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
    
    return output_buffer.getvalue()