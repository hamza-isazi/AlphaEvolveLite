
import reedsolo

def encode(text: str) -> bytes:
    """Encode a string with Reed–Solomon, return bytes."""
    return reedsolo.RSCodec().encode(text.encode('utf-8'))


def decode(data: bytes) -> str:
    """Decode Reed–Solomon bytes, return string."""
    decoded_bytes, _, _ = reedsolo.RSCodec().decode(data)
    return decoded_bytes.decode('utf-8')


if __name__ == "__main__":
    message = "Hello, world!"
    print(encode(message))
    print(decode(encode(message)))
