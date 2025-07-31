def compress(text: str) -> bytes:
    # Naive run-length encoding (RLE) for ASCII text
    if not text:
        return b""
    
    output = []
    prev_char = text[0]
    count = 1
    for c in text[1:]:
        if c == prev_char and count < 255:
            count += 1
        else:
            output.append(bytes([count]))
            output.append(prev_char.encode("latin1"))
            prev_char = c
            count = 1
    output.append(bytes([count]))
    output.append(prev_char.encode("latin1"))
    return b"".join(output)

def decompress(data: bytes) -> str:
    # Decode RLE format produced above
    output = []
    for i in range(0, len(data), 2):
        count = data[i]
        char = data[i + 1:i + 2].decode("latin1")
        output.append(char * count)
    return "".join(output)
