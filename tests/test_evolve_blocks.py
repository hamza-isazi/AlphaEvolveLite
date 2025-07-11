#!/usr/bin/env python
"""
Test script to verify evolve block detection and conditional behavior.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import alphaevolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphaevolve.patcher import PatchApplier, _EVOLVE_RE
from alphaevolve.prompts import PromptSampler

def test_evolve_block_detection():
    """Test that evolve blocks are correctly detected."""
    print("Testing evolve block detection...")
    
    # Code with evolve blocks
    code_with_blocks = """
def test_function():
    # EVOLVE-BLOCK-START
    x = 1
    y = 2
    # EVOLVE-BLOCK-END
    return x + y
"""
    
    # Code without evolve blocks
    code_without_blocks = """
def test_function():
    x = 1
    y = 2
    return x + y
"""
    
    # Test patcher
    regions_with = PatchApplier._evolve_regions(code_with_blocks)
    regions_without = PatchApplier._evolve_regions(code_without_blocks)
    
    print(f"  Code with evolve blocks: {len(regions_with)} regions found")
    print(f"  Code without evolve blocks: {len(regions_without)} regions found")
    
    # Test prompt sampler
    sampler = PromptSampler(None)
    has_blocks = sampler._has_evolve_blocks(code_with_blocks)
    has_no_blocks = sampler._has_evolve_blocks(code_without_blocks)
    
    print(f"  PromptSampler detects blocks: {has_blocks}")
    print(f"  PromptSampler detects no blocks: {has_no_blocks}")
    
    assert len(regions_with) > 0, "Should find evolve regions in code with blocks"
    assert len(regions_without) == 0, "Should find no evolve regions in code without blocks"
    assert has_blocks, "PromptSampler should detect evolve blocks"
    assert not has_no_blocks, "PromptSampler should not detect evolve blocks"
    
    print("✓ Evolve block detection tests passed!")

def test_patch_application():
    """Test that patches are applied correctly with and without evolve blocks."""
    print("\nTesting patch application...")
    
    # Code with evolve blocks
    code_with_blocks = """
def test_function():
    # EVOLVE-BLOCK-START
    x = 1
    y = 2
    # EVOLVE-BLOCK-END
    return x + y
"""
    
    # Code without evolve blocks
    code_without_blocks = """
def test_function():
    x = 1
    y = 2
    return x + y
"""
    
    # Valid diff that should work in both cases
    valid_diff = """<<<<<<< SEARCH
    x = 1
    y = 2
=======
    x = 10
    y = 20
>>>>>>> REPLACE"""
    
    # Test with evolve blocks
    result_with = PatchApplier.apply_diff(code_with_blocks, valid_diff)
    print(f"  Patch with evolve blocks: {'✓ Applied' if result_with else '✗ Failed'}")
    
    # Test without evolve blocks
    result_without = PatchApplier.apply_diff(code_without_blocks, valid_diff)
    print(f"  Patch without evolve blocks: {'✓ Applied' if result_without else '✗ Failed'}")
    
    # Test invalid diff (outside evolve blocks when they exist)
    invalid_diff = """<<<<<<< SEARCH
def test_function():
=======
def new_function():
>>>>>>> REPLACE"""
    
    result_invalid = PatchApplier.apply_diff(code_with_blocks, invalid_diff)
    print(f"  Invalid patch with evolve blocks: {'✗ Rejected' if not result_invalid else '✓ Applied (unexpected)'}")
    
    # Test invalid diff without evolve blocks (should work)
    result_invalid_no_blocks = PatchApplier.apply_diff(code_without_blocks, invalid_diff)
    print(f"  Invalid patch without evolve blocks: {'✓ Applied' if result_invalid_no_blocks else '✗ Failed'}")
    
    assert result_with is not None, "Valid patch should be applied when evolve blocks exist"
    assert result_without is not None, "Valid patch should be applied when no evolve blocks exist"
    assert result_invalid is None, "Invalid patch should be rejected when evolve blocks exist"
    assert result_invalid_no_blocks is not None, "Any patch should work when no evolve blocks exist"
    
    print("✓ Patch application tests passed!")

def test_prompt_generation():
    """Test that prompts are generated correctly based on evolve block presence."""
    print("\nTesting prompt generation...")
    
    # Mock parent row
    parent_with_blocks = {
        'code': """
def test_function():
    # EVOLVE-BLOCK-START
    x = 1
    y = 2
    # EVOLVE-BLOCK-END
    return x + y
""",
        'score': 1.0
    }
    
    parent_without_blocks = {
        'code': """
def test_function():
    x = 1
    y = 2
    return x + y
""",
        'score': 1.0
    }
    
    sampler = PromptSampler(None)
    
    # Generate prompts
    prompt_with = sampler.build(parent_with_blocks, [])
    prompt_without = sampler.build(parent_without_blocks, [])
    
    # Check that the appropriate template was used
    has_evolve_instructions = "Only change lines *between* the markers" in prompt_with
    has_free_instructions = "You can modify any part of the code as needed" in prompt_without
    
    print(f"  Prompt with evolve blocks contains evolve instructions: {has_evolve_instructions}")
    print(f"  Prompt without evolve blocks contains free instructions: {has_free_instructions}")
    
    assert has_evolve_instructions, "Prompt with evolve blocks should contain evolve instructions"
    assert has_free_instructions, "Prompt without evolve blocks should contain free instructions"
    
    print("✓ Prompt generation tests passed!")

if __name__ == "__main__":
    print("Running evolve block tests...\n")
    
    try:
        test_evolve_block_detection()
        test_patch_application()
        test_prompt_generation()
        print("\n🎉 All tests passed! The evolve block conditional behavior is working correctly.")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 