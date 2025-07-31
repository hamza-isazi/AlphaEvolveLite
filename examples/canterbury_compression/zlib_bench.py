import zlib

def compress(text: str) -> bytes:
    return zlib.compress(text.encode("latin1"))

def decompress(data: bytes) -> str:
    return zlib.decompress(data).decode("latin1")