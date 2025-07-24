from io import StringIO

def main(input_text: str) -> str:
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

    # Build output string directly instead of capturing stdout
    output_lines = []
    
    # Naive strategy: sign up all libraries in input order, scan all books
    output_lines.append(str(L))
    for i, (T, M, books) in enumerate(libraries):
        books = books[:min(len(books), (D - T) * M)]
        output_lines.append(f"{i} {len(books)}")
        output_lines.append(" ".join(map(str, books)))
    
    return "\n".join(output_lines)