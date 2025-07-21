#!/usr/bin/env python
"""
Test script to verify indentation flexibility in diff application.
"""

import sys
from pathlib import Path
import unittest
from alphaevolve.patcher import PatchApplier

# Add the parent directory to the path so we can import alphaevolve
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_diff_test_case(name, original_code, diff, expected):
    """Helper to run a diff test case and check indentation and correctness."""
    print(f"\nTesting case: {name}...")
    result = PatchApplier.apply_diff(original_code, diff)
    if result is not None:
        print("✓ Diff was applied successfully")
        if result == expected:
            print("✓ Indentation preserved correctly")
            return True
        else:
            print("✗ Indentation was not preserved correctly")
            # Print specific lines where there were mismatches
            result_lines = result.splitlines()
            expected_lines = expected.splitlines()
            max_len = max(len(result_lines), len(expected_lines))
            print("Line-by-line diff (expected vs result):")
            for i in range(max_len):
                exp_line = expected_lines[i] if i < len(expected_lines) else "<no line>"
                res_line = result_lines[i] if i < len(result_lines) else "<no line>"
                if exp_line != res_line:
                    print(f"Line {i+1}:")
                    print(f"  Expected: {repr(exp_line)}")
                    print(f"  Result:   {repr(res_line)}")
            return False
    else:
        print("✗ Diff failed to apply")
        return False

def test_indentation_cases():
    """Test the patching with various indentation cases both in the original code and in the diff text.
    We want the patcher to work both when the LLM uses the absolute indentation from the source code and
    when it uses a relative indentation, meaning the diff text has a base indentation level of 0."""
    original_code = """try:
    # Create a StringIO object for input
    input_buffer = StringIO(input_text)
    
    def readints():
        return list(map(int, input_buffer.readline().split()))

    B, L, D = readints()
    scores = readints()
    from heapq import heappush, heappop

    libraries = []
    for idx in range(L):
        N, T, M = readints()
        books = readints()
        total_book_score = sum(scores[b] for b in books)
        libraries.append((T, M, total_book_score, books, idx))

    # Prioritize libraries based on a heuristic score
    libraries.sort(key=lambda lib: (lib[0], -lib[2] / lib[0], -lib[1]))
except Exception:
    pass"""
    diff = """<<<<<< SEARCH
libraries.append((T, M, total_book_score, books, idx))
=======
books_with_scores = [(b, scores[b]) for b in books]
libraries.append((T, M, total_book_score, books_with_scores, idx))
if(True):
    print("hello")
>>>>>>> REPLACE"""
    expected = """try:
    # Create a StringIO object for input
    input_buffer = StringIO(input_text)

    def readints():
        return list(map(int, input_buffer.readline().split()))

    B, L, D = readints()
    scores = readints()
    from heapq import heappush, heappop

    libraries = []
    for idx in range(L):
        N, T, M = readints()
        books = readints()
        total_book_score = sum(scores[b] for b in books)
        books_with_scores = [(b, scores[b]) for b in books]
        libraries.append((T, M, total_book_score, books_with_scores, idx))
        if(True):
            print("hello")

    # Prioritize libraries based on a heuristic score
    libraries.sort(key=lambda lib: (lib[0], -lib[2] / lib[0], -lib[1]))
except Exception:
    pass"""
        
    if run_diff_test_case("relative indentation in diff", original_code, diff, expected) == False:
        return False
    
    original_code = """        # Improved strategy: prioritize libraries by signup time and potential score
        library_scores = []
        for i, (T, M, books) in enumerate(libraries):
            potential_score = sum(scores[book] for book in books)
            library_scores.append((i, T, potential_score, M, books))

        # Sort libraries by a combination of signup time, potential score per day, and daily scan rate
        library_scores.sort(key=lambda x: (x[1], -x[2]/x[1], -x[3]))"""
    diff = """<<<<<<< SEARCH
            potential_score = sum(scores[book] for book in books)
            library_scores.append((i, T, potential_score, M, books))

        # Sort libraries by a combination of signup time, potential score per day, and daily scan rate
        library_scores.sort(key=lambda x: (x[1], -x[2]/x[1], -x[3]))
=======
            unique_books = set(books) - scanned_books
            potential_score = sum(scores[book] for book in unique_books)
            library_scores.append((i, T, potential_score, M, unique_books))

        # Sort libraries by a combination of adjusted potential score, signup time, and daily scan rate
        library_scores.sort(key=lambda x: (-x[2]/x[1], x[1], -x[3]))
>>>>>>> REPLACE"""
    expected = """        # Improved strategy: prioritize libraries by signup time and potential score
        library_scores = []
        for i, (T, M, books) in enumerate(libraries):
            unique_books = set(books) - scanned_books
            potential_score = sum(scores[book] for book in unique_books)
            library_scores.append((i, T, potential_score, M, unique_books))

        # Sort libraries by a combination of adjusted potential score, signup time, and daily scan rate
        library_scores.sort(key=lambda x: (-x[2]/x[1], x[1], -x[3]))"""
    
    if run_diff_test_case("absolute indentation in diff", original_code, diff, expected) == False:
        return False

    original_code = """        libraries = []
        for _ in range(L):
            N, T, M = readints()
            books = readints()
            libraries.append((T, M, books))

        # Naive strategy: sign up all libraries in input order, scan all books
        print(L)"""
    diff = """<<<<<<< SEARCH
        libraries = []
        for _ in range(L):
            N, T, M = readints()
            books = readints()
            libraries.append((T, M, books))

        # Naive strategy: sign up all libraries in input order, scan all books
        print(L)
=======
        libraries = []
        for _ in range(L):
            N, T, M = readints()
            books = readints()
            libraries.append((T, M, books))
        
        # Naive strategy: sign up all libraries in input order, scan all books
        print(L)
        print(libraries)
>>>>>>> REPLACE"""
    expected = """        libraries = []
        for _ in range(L):
            N, T, M = readints()
            books = readints()
            libraries.append((T, M, books))

        # Naive strategy: sign up all libraries in input order, scan all books
        print(L)
        print(libraries)"""
    
    if run_diff_test_case("empty lines with whitespace in diff", original_code, diff, expected) == False:
        return False

    return True

class TestPatchApplier(unittest.TestCase):
    
    def test_extract_code_from_markdown(self):
        """Test code extraction from markdown code blocks with various formats."""
        
        # Test basic code block
        text1 = "```python\ndef hello():\n    print('Hello')\n```"
        result1 = PatchApplier._extract_code_from_markdown(text1)
        expected1 = "def hello():\n    print('Hello')"
        self.assertEqual(result1, expected1)
        
        # Test code block with text after closing backticks
        text2 = "```python\ndef hello():\n    print('Hello')\n```\n\nThis complete file replacement introduces a more effective approach."
        result2 = PatchApplier._extract_code_from_markdown(text2)
        expected2 = "def hello():\n    print('Hello')"
        self.assertEqual(result2, expected2)
        
        # Test code block without language identifier
        text3 = "```\ndef hello():\n    print('Hello')\n```"
        result3 = PatchApplier._extract_code_from_markdown(text3)
        expected3 = "def hello():\n    print('Hello')"
        self.assertEqual(result3, expected3)
        
        # Test text without code blocks (should return original)
        text4 = "This is just regular text without code blocks"
        result4 = PatchApplier._extract_code_from_markdown(text4)
        self.assertEqual(result4, text4)
        
        # Test code block with complex language identifier
        text5 = "```python3.9\nimport sys\nprint(sys.version)\n```"
        result5 = PatchApplier._extract_code_from_markdown(text5)
        expected5 = "import sys\nprint(sys.version)"
        self.assertEqual(result5, expected5)

        # Test code block with last backticks on the same line
        text6 = "```python\ndef hello():\n    print('Hello')```"
        result6 = PatchApplier._extract_code_from_markdown(text6)
        expected6 = "def hello():\n    print('Hello')"
        self.assertEqual(result6, expected6)

if __name__ == "__main__":
    print("Running indentation flexibility tests...")
    
    success = True
    success &= test_indentation_cases()
    test_patcher = TestPatchApplier()
    test_patcher.test_extract_code_from_markdown()
    
    if success:
        print("\n🎉 All tests passed! The indentation flexibility fix is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 