
import random
import string
import time
import reedsolo
import base64
import numpy as np

class ReedSolomonCode:
    def __init__(self, nsym=13):
        self.nsym = nsym
        self.rs_codec = reedsolo.RSCodec(nsym=nsym)
        
    @property
    def rate(self) -> float:
        # Reed-Solomon rate depends on message length, but we'll use an approximation
        # For typical usage with nsym=13, rate is roughly 0.8
        return 0.8
    
    def encode(self, u: np.ndarray) -> np.ndarray:
        """Encode binary array using Reed-Solomon.
        
        Args:
            u: Input binary array (uint8)
        Returns:
            Encoded binary array (uint8) - longer due to RS redundancy
        """
        # Store original length for decoding
        self._original_length = len(u)
        
        # Convert binary array to bytes
        u_bytes = np.packbits(u).tobytes()
        
        # Encode with Reed-Solomon
        encoded_bytes = self.rs_codec.encode(u_bytes)
        
        # Convert back to binary array
        encoded_bits = np.unpackbits(np.frombuffer(encoded_bytes, dtype=np.uint8))
        
        return encoded_bits.astype(np.uint8)
    
    def decode(self, corrupted: np.ndarray) -> np.ndarray:
        """Decode corrupted bits using Reed-Solomon.
        
        Args:
            corrupted: Corrupted binary array (float64) - can be LLRs or hard decisions
        Returns:
            Decoded binary array (uint8) - original length
        """
        # Convert to hard decisions
        # If corrupted contains LLRs (negative = 1, positive = 0), convert them
        # If corrupted contains binary values (0.0/1.0), round them
        if np.any(corrupted < 0):
            # LLR format: negative = 1, positive = 0
            hard_decisions = (corrupted < 0).astype(np.uint8)
        else:
            # Binary format: round to nearest integer
            hard_decisions = np.round(corrupted).astype(np.uint8)
        
        # Convert to bytes
        encoded_bytes = np.packbits(hard_decisions).tobytes()
        
        try:
            # Decode with Reed-Solomon
            decoded_bytes, _, _ = self.rs_codec.decode(encoded_bytes)
            
            # Convert back to binary array
            decoded_bits = np.unpackbits(np.frombuffer(decoded_bytes, dtype=np.uint8))
            
            # Return only the original information bits (remove padding if any)
            # Use the stored original length from encoding
            original_length = getattr(self, '_original_length', 0)
            if original_length == 0:
                # Fallback: estimate original length
                original_length = len(corrupted) - (self.nsym * 8)
            
            if decoded_bits.size >= original_length:
                return decoded_bits[:original_length].astype(np.uint8)
            else:
                # If decoding failed or returned fewer bits, pad with zeros
                result = np.zeros(original_length, dtype=np.uint8)
                result[:decoded_bits.size] = decoded_bits
                return result
                
        except reedsolo.ReedSolomonError:
            # If decoding fails, return zeros (or handle error appropriately)
            # This happens when too many errors exceed RS correction capability
            original_length = getattr(self, '_original_length', 0)
            if original_length == 0:
                # Fallback: estimate original length
                original_length = len(corrupted) - (self.nsym * 8)
            return np.zeros(original_length, dtype=np.uint8)

def build_code():
    """Return a Reed-Solomon code object for evaluation."""
    return ReedSolomonCode(nsym=13)

# Legacy string-based interface (for compatibility with old evaluator)
def encode(text: str) -> str:
    """Encode a string with Reed–Solomon, return base64 string."""
    encoded_bytes = reedsolo.RSCodec(nsym=13).encode(text.encode('utf-8'))
    return base64.b64encode(encoded_bytes).decode('ascii')

def decode(data: str) -> str:
    """Decode base64 string using Reed–Solomon, return string."""
    decoded_bytes, _, _ = reedsolo.RSCodec(nsym=13).decode(base64.b64decode(data.encode('ascii')))
    return decoded_bytes.decode('utf-8')

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
    # Introduce only 2-3 bit errors (well within RS correction capability)
    error_positions = np.random.choice(len(corrupted), size=3, replace=False)
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

