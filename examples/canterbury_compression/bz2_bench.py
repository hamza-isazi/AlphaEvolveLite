import bz2

def compress(text: str) -> bytes:
    return bz2.compress(text.encode("latin1"))

def decompress(data: bytes) -> str:
    return bz2.decompress(data).decode("latin1")