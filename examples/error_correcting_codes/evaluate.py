import json
import os
import uuid
import importlib.util
from typing import Dict, List, Tuple
import numpy as np

# -------------------------
# Symbol Error Simulation + utilities
# -------------------------
def introduce_symbol_errors(data: np.ndarray, error_rate: float, rng: np.random.RandomState) -> np.ndarray:
    """Introduce random bit flips (symbol errors) into binary data.
    
    Args:
        data: Input binary array
        error_rate: Probability of each bit being flipped (0.0 to 1.0)
        rng: Random number generator for reproducibility
    Returns:
        Data with random bit flips
    """
    corrupted = data.copy()
    # Generate random bit flips based on error rate
    flip_mask = rng.random(data.shape) < error_rate
    corrupted[flip_mask] = 1 - corrupted[flip_mask]  # Flip bits
    return corrupted

def introduce_byte_errors(data: np.ndarray, error_rate: float, rng: np.random.RandomState) -> np.ndarray:
    """Introduce byte-level errors (more suitable for Reed-Solomon).
    
    Args:
        data: Input binary array
        error_rate: Probability of each byte (8 bits) being corrupted
        rng: Random number generator for reproducibility
    Returns:
        Data with random byte corruptions
    """
    corrupted = data.copy()
    # Convert to bytes, introduce errors, then back to bits
    data_bytes = np.packbits(data)
    
    # Introduce errors at byte level
    byte_error_mask = rng.random(len(data_bytes)) < error_rate
    for i in range(len(data_bytes)):
        if byte_error_mask[i]:
            # Corrupt entire byte by flipping some bits
            data_bytes[i] ^= rng.randint(1, 256)
    
    # Convert back to bits
    corrupted = np.unpackbits(data_bytes)
    
    # Ensure same length as input
    if len(corrupted) < len(data):
        corrupted = np.concatenate([corrupted, np.zeros(len(data) - len(corrupted), dtype=np.uint8)])
    elif len(corrupted) > len(data):
        corrupted = corrupted[:len(data)]
    
    return corrupted

def error_rate_to_snr_db(error_rate: float) -> float:
    """Convert bit error rate to equivalent SNR in dB for reference.
    This is approximate: BER ≈ Q(sqrt(2*SNR)) for BPSK in AWGN.
    """
    if error_rate <= 0:
        return float('inf')
    elif error_rate >= 0.5:
        return -float('inf')
    else:
        # Approximate inverse of Q function
        # For small BER: SNR ≈ -10*log10(2*BER)
        return -10 * np.log10(2 * error_rate)

class CommonRNG:
    """Reuses data across candidates to reduce variance."""
    def __init__(self, error_rates: List[float], frame_len_info: int, max_frames: int, seed: int = 1234):
        self.error_rates = error_rates
        self.frame_len_info = frame_len_info
        self.max_frames = max_frames
        self.rs = np.random.RandomState(seed)
        self.data = {}
        for err_rate in error_rates:
            u_frames = self.rs.randint(0, 2, size=(max_frames, frame_len_info), dtype=np.uint8)
            self.data[err_rate] = {'u': u_frames}

# -------------------------
# Evaluator
# -------------------------
class Evaluator:
    """
    Reliable throughput at target error rate over different error rate levels.
    Two-stage sampling with early stops and common RNG.
    """
    def __init__(
        self,
        error_rates=(0.0001, 0.0005, 0.001, 0.002),  # Very low bit error rates for RS
        target_error_rate=1e-3,  # Target frame error rate
        screen_frames=400,
        promote_frames=5000,
        max_frames=6000,
        info_len_target=250,
        seed=1234,
    ):
        self.error_rates = list(error_rates)
        self.target_error_rate = target_error_rate
        self.screen_frames = screen_frames
        self.promote_frames = promote_frames
        self.max_frames = max_frames
        self.info_len_target = info_len_target
        self.rng = CommonRNG(self.error_rates, info_len_target, max_frames, seed)

    def _simulate_frames(self, code_obj, error_rate: float, n_frames: int) -> Tuple[int, int]:
        """Return (frame_errors, total_frames). Expects code_obj.encode() and code_obj.decode()."""
        u_all = self.rng.data[error_rate]['u'][:n_frames]
        frame_errs = 0
        
        for u in u_all:
            # Encode the data
            c = code_obj.encode(u)
            
            # Choose error introduction method based on code type
            # Reed-Solomon works better with byte-level errors
            if hasattr(code_obj, 'rs_codec'):  # Reed-Solomon
                corrupted = introduce_byte_errors(c, error_rate, self.rng.rs)
            else:  # Convolutional or other codes
                corrupted = introduce_symbol_errors(c, error_rate, self.rng.rs)
            
            # Decode the corrupted data
            # For symbol error simulation, we pass the corrupted bits directly
            # (no need for LLRs since we're doing hard-decision decoding)
            u_hat = code_obj.decode(corrupted.astype(np.float64))  # Convert to float for compatibility
            
            # Check if decoding was successful
            if u_hat.size != u.size or np.any(u_hat != u):
                frame_errs += 1
                
        return frame_errs, n_frames

    def evaluate_code(self, code_obj) -> Dict:
        """Returns dict with 'score' and per-error-rate stats. Requires code_obj.rate (float)."""
        per_error_rate = {}
        for err_rate in self.error_rates:
            # Stage 1: screen
            fe, n = self._simulate_frames(code_obj, err_rate, self.screen_frames)
            fer = fe / n
            if fe == 0:
                per_error_rate[err_rate] = {'frames': n, 'frame_errors': fe, 'FER': fer, 'passed': True}
                continue
            max_allowed = int(np.floor(self.target_error_rate * self.promote_frames))
            if fe > max_allowed:
                per_error_rate[err_rate] = {'frames': n, 'frame_errors': fe, 'FER': fer, 'passed': False}
                continue
            # Stage 2: promote
            add = self.promote_frames - n
            fe2, n2 = self._simulate_frames(code_obj, err_rate, add)
            fe_total = fe + fe2
            n_total = n + n2
            fer2 = fe_total / n_total
            passed = fe_total <= max_allowed
            per_error_rate[err_rate] = {'frames': n_total, 'frame_errors': fe_total, 'FER': fer2, 'passed': bool(passed)}

        score = np.mean([code_obj.rate if per_error_rate[err]['passed'] else 0.0 for err in self.error_rates])
        return {
            'score': float(score),
            'rate': float(code_obj.rate),
            'error_rates': self.error_rates,
            'target_error_rate': self.target_error_rate,
            'per_error_rate': per_error_rate
        }

# -------------------------
# Public API
# -------------------------
def _load_program(program_path: str):
    """Load a Python module from file path, returning the module object."""
    module_name = f"ae_prog_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {program_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def evaluate(program_path: str) -> float:
    """
    Load candidate program and return its scalar fitness:
    Reliable throughput (mean over error rates) at target frame error rate.
    The candidate module must expose build_code() -> object with:
        - encode(u: np.ndarray[uint8]) -> np.ndarray[uint8]  (coded bits)
        - decode(corrupted: np.ndarray[float64]) -> np.ndarray[uint8] (decoded info bits)
        - rate: float
    """
    mod = _load_program(program_path)
    if not hasattr(mod, "build_code"):
        raise AttributeError("Program must define build_code()")
    code_obj = mod.build_code()
    for attr in ("encode", "decode", "rate"):
        if not hasattr(code_obj, attr):
            raise AttributeError("Code object must provide encode(), decode(), and rate")
    evaluator = Evaluator()
    res = evaluator.evaluate_code(code_obj)
    return float(res["score"])

# Optional CLI for quick checks
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("program_path", help="Path to candidate program (e.g., initial_program.py)")
    p.add_argument("--json", action="store_true", help="Print full JSON result instead of just score")
    args = p.parse_args()

    mod = _load_program(args.program_path)
    code_obj = mod.build_code()
    evaluator = Evaluator()
    result = evaluator.evaluate_code(code_obj)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result["score"])