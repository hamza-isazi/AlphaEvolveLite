import lzma

def compress(text: str) -> bytes:
    return lzma.compress(text.encode("latin1"), preset=9)

def decompress(data: bytes) -> str:
    return lzma.decompress(data).decode("latin1")
