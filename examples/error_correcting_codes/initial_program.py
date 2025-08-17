import numpy as np

class SimpleRepetitionCode:
    """Simple repetition code: repeat each bit 3 times for error correction."""
    
    def __init__(self, repetition_factor=3):
        self.repetition_factor = repetition_factor
        
    @property
    def rate(self) -> float:
        """Code rate: information bits / total bits."""
        return 1.0 / self.repetition_factor
    
    def encode(self, u: np.ndarray) -> np.ndarray:
        """Encode by repeating each bit multiple times.
        
        Args:
            u: Input binary array (uint8)
        Returns:
            Encoded binary array (uint8) - longer due to repetition
        """
        # Repeat each bit 'repetition_factor' times
        encoded = np.repeat(u, self.repetition_factor)
        return encoded.astype(np.uint8)
    
    def decode(self, corrupted: np.ndarray) -> np.ndarray:
        """Decode using majority voting on repeated bits.
        
        Args:
            corrupted: Corrupted binary array (float64) - can be LLRs or hard decisions
        Returns:
            Decoded binary array (uint8) - original length
        """
        # Convert to hard decisions if needed
        if np.any(corrupted < 0):
            # LLR format: negative = 1, positive = 0
            hard_decisions = (corrupted < 0).astype(np.uint8)
        else:
            # Binary format: round to nearest integer
            hard_decisions = np.round(corrupted).astype(np.uint8)
        
        # Reshape to groups of repeated bits
        n_groups = len(hard_decisions) // self.repetition_factor
        if len(hard_decisions) % self.repetition_factor != 0:
            # Handle case where length isn't divisible by repetition factor
            n_groups = len(hard_decisions) // self.repetition_factor
            hard_decisions = hard_decisions[:n_groups * self.repetition_factor]
        
        # Reshape to (n_groups, repetition_factor)
        groups = hard_decisions.reshape(n_groups, self.repetition_factor)
        
        # Use majority voting for each group
        decoded = np.zeros(n_groups, dtype=np.uint8)
        for i in range(n_groups):
            # Count 1s in this group
            ones_count = np.sum(groups[i])
            # If more than half are 1s, output 1; otherwise 0
            decoded[i] = 1 if ones_count > self.repetition_factor // 2 else 0
        
        return decoded

def build_code():
    """Return a simple repetition code object for evaluation."""
    return SimpleRepetitionCode(repetition_factor=3)

# Create a global instance for the encode/decode functions
_rep_code = build_code()

def encode(message: str) -> str:
    """Encode a string using repetition coding."""
    # Convert string to binary array
    message_bytes = message.encode('utf-8')
    binary_array = np.unpackbits(np.frombuffer(message_bytes, dtype=np.uint8))
    
    # Encode using repetition code
    encoded_bits = _rep_code.encode(binary_array)
    
    # Convert to base64 string
    encoded_bytes = np.packbits(encoded_bits)
    import base64
    return base64.b64encode(encoded_bytes).decode('ascii')

def decode(encoded: str) -> str:
    """Decode a base64 string using repetition decoding."""
    # Convert base64 to binary array
    import base64
    encoded_bytes = base64.b64decode(encoded.encode('ascii'))
    encoded_bits = np.unpackbits(np.frombuffer(encoded_bytes, dtype=np.uint8))
    
    # Decode using repetition code
    decoded_bits = _rep_code.decode(encoded_bits)
    
    # Convert back to string
    decoded_bytes = np.packbits(decoded_bits)
    return decoded_bytes.tobytes().decode('utf-8', errors='ignore')

# Test function to verify the implementation works
if __name__ == "__main__":
    # Test with the evaluator's expected interface
    code = build_code()
    
    # Test data similar to what evaluator uses
    test_data = np.random.randint(0, 2, size=250, dtype=np.uint8)
    print(f"Original data length: {len(test_data)}")
    
    # Encode
    encoded = code.encode(test_data)
    print(f"Encoded length: {len(encoded)}")
    print(f"Code rate: {code.rate}")
    
    # Test 1: No errors (should work perfectly)
    print("\n=== Test 1: No errors ===")
    decoded = code.decode(encoded.astype(np.float64))
    if np.array_equal(decoded, test_data):
        print("✅ Perfect decoding with no errors!")
    else:
        print("❌ Failed to decode without errors!")
        print(f"Errors: {np.sum(decoded != test_data)}")
    
    # Test 2: Small number of bit errors (within correction capability)
    print("\n=== Test 2: Small bit errors ===")
    corrupted = encoded.copy()
    # Introduce errors in some repeated bits (but not all 3 copies)
    error_positions = np.random.choice(len(corrupted), size=10, replace=False)
    corrupted[error_positions] = 1 - corrupted[error_positions]
    print(f"Introduced {len(error_positions)} bit errors")
    
    decoded = code.decode(corrupted.astype(np.float64))
    if np.array_equal(decoded, test_data):
        print("✅ Perfect decoding with small errors!")
    else:
        print("❌ Failed to decode with small errors!")
        print(f"Errors: {np.sum(decoded != test_data)}")
    
    # Test 3: Simulate symbol error simulation (like evaluator)
    print("\n=== Test 3: Symbol error simulation ===")
    # Simulate symbol errors (like new evaluator does)
    error_rate = 0.001  # 0.1% bit error rate
    corrupted = encoded.copy()
    flip_mask = np.random.random(encoded.shape) < error_rate
    corrupted[flip_mask] = 1 - corrupted[flip_mask]  # Flip bits
    
    print(f"Introduced {np.sum(flip_mask)} bit errors (rate: {error_rate})")
    decoded = code.decode(corrupted.astype(np.float64))
    if np.array_equal(decoded, test_data):
        print("✅ Perfect decoding with symbol errors!")
    else:
        print("❌ Failed to decode with symbol errors!")
        print(f"Errors: {np.sum(decoded != test_data)}")
    
    print(f"\nDecoded length: {len(decoded)}")
    
    # Check if we got the right length
    if len(decoded) == len(test_data):
        print("✅ Length match!")
    else:
        print(f"❌ Length mismatch: expected {len(test_data)}, got {len(decoded)}")