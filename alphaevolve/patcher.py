import re
from typing import Optional, List, Tuple

_DIFF_RE = re.compile(
    r"<<{4,}\s*SEARCH\s*\n(.*?)\n={4,}\s*\n(.*?)\n>{4,}\s*REPLACE",
    re.DOTALL,
)
_EVOLVE_RE = re.compile(
    r"#\s*EVOLVE-BLOCK-START(.*?)#\s*EVOLVE-BLOCK-END",
    re.DOTALL,
)
_CODE_BLOCK_RE = re.compile(
    r"```(?:[a-zA-Z0-9_+.-]*)\n(.*?)\n```",
    re.DOTALL,
)

class PatchError(Exception):
    """Exception raised when patch application fails."""
    pass

class PatchApplier:
    @staticmethod
    def _extract_code_from_markdown(text: str) -> str:
        """
        Extract code from markdown code blocks, handling various formats.
        
        This method properly handles code blocks that may have:
        - Opening ``` with optional language identifier
        - Closing ``` that may be followed by additional text
        - Multiple code blocks (takes the first complete one)
        
        Args:
            text: Text that may contain markdown code blocks
            
        Returns:
            The extracted code without markdown formatting
        """
        # Look for code block pattern: ```[language]?\n...\n```
        # The closing ``` should be followed by a newline or end of string
        match = _CODE_BLOCK_RE.search(text)
        
        if match:
            return match.group(1)
        else:
            raise PatchError(f"No code block found in response: {repr(text[:100])}...")

    @staticmethod
    def _remove_right_whitespace(text: str) -> str:
        """
        Remove trailing whitespace from each line while preserving leading whitespace.
        
        This sanitizes text by removing spaces and tabs at the end of each line,
        which can cause issues with diff matching while keeping indentation intact.
        
        Args:
            text: The text to sanitize
            
        Returns:
            Text with trailing whitespace removed from each line
        """
        lines = text.split('\n')
        sanitized_lines = [line.rstrip() for line in lines]
        return '\n'.join(sanitized_lines)

    @staticmethod
    def _evolve_regions(src: str) -> List[Tuple[int, int]]:
        return [(m.start(1), m.end(1)) for m in _EVOLVE_RE.finditer(src)]

    @staticmethod
    def _normalize_indentation(text: str) -> str:
        """Normalize indentation by removing leading whitespace from each line."""
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                normalized_lines.append(line.lstrip())
            else:  # Empty line
                normalized_lines.append('')
        return '\n'.join(normalized_lines)

    @staticmethod
    def _preserve_relative_indentation(original_matched: str, replace: str) -> str:
        """
        Preserve the relative indentation structure of replacement lines while applying base indentation.
        
        This function ensures that nested code structures (like if/else blocks, function definitions,
        or loops) maintain their proper indentation levels relative to each other, while aligning
        the entire block to match the context where it's being inserted.
        
        Args:
            original_matched: The original matched text to determine base indentation
            replace: The replacement text
            
        Returns:
            A string with the replacement lines properly indented
        """
        original_lines = original_matched.split('\n')

        # If the original had no indentation, return replacement as-is
        if not original_lines or not original_lines[0].startswith(' '):
            return replace
                
        # Find the minimum indentation in original lines
        min_indent_original = float('inf')
        for line in original_lines:
            if line.strip():  # Only consider non-empty lines
                indent_level = len(line) - len(line.lstrip())
                min_indent_original = min(min_indent_original, indent_level)
        
        if min_indent_original == float('inf'):
            min_indent_original = 0  # No non-empty lines
        
        base_indent = ' ' * int(min_indent_original)
        replacement_lines = replace.split('\n')
        adjusted_replacement_lines = []
        
        # Apply indentation preserving relative structure
        for line in replacement_lines:
            if not line.lstrip():  # Blank line (empty or only spaces)
                adjusted_replacement_lines.append('')
            else:
                # Calculate relative indentation for the replacement lines
                relative_indent = len(line) - len(line.lstrip())
                # Apply base indentation plus relative indentation
                adjusted_line = base_indent + (' ' * int(relative_indent)) + line.lstrip()
                adjusted_replacement_lines.append(adjusted_line)
        
        return '\n'.join(adjusted_replacement_lines)

    @staticmethod
    def _find_with_indentation_flexibility(source: str, search_text: str) -> Optional[Tuple[int, int, str]]:
        """
        Find search_text in source with flexible indentation matching.
        
        Returns:
            Tuple of (start_pos, end_pos, original_matched_text) or None if not found
        """
        # Normalize the search text
        normalized_search = PatchApplier._normalize_indentation(search_text)
        search_lines = search_text.split('\n')
        
        # Split source into lines and check each possible starting position
        source_lines = source.split('\n')
        
        for i in range(len(source_lines) - len(search_lines) + 1):
            # Extract candidate block and normalize it
            candidate_lines = source_lines[i:i + len(search_lines)]
            candidate_text = '\n'.join(candidate_lines)
            normalized_candidate = PatchApplier._normalize_indentation(candidate_text)
            
            # Compare normalized versions
            if normalized_candidate == normalized_search:
                # Found the match! Calculate positions
                start_pos = len('\n'.join(source_lines[:i])) + (1 if i > 0 else 0)
                end_pos = len('\n'.join(source_lines[:i + len(search_lines)]))
                return start_pos, end_pos, candidate_text
        
        return None

    @staticmethod
    def apply_diff(source: str, diff_text: str) -> str:
        # Sanitize inputs by removing trailing whitespace while preserving indentation
        source = PatchApplier._remove_right_whitespace(source)
        diff_text = PatchApplier._remove_right_whitespace(diff_text)
        
        diffs = _DIFF_RE.findall(diff_text)
        # If no diffs (SEARCH/REPLACE blocks) are present, use full file replacement
        if not diffs:
            return PatchApplier._extract_code_from_markdown(diff_text)
        
        # Handle SEARCH/REPLACE format
        regions = PatchApplier._evolve_regions(source)
        new = source
        for search, replace in diffs:            
            # Determine if the LLM is using relative indentation by checking if the first line is indented
            # If it is, we need to preserve the relative indentation structure of the replacement lines
            # while applying base indentation.
            search_lines = search.split('\n')
            first_line = search_lines[0] if search_lines else ''
            is_first_line_indented = first_line.startswith(' ') or first_line.startswith('\t')

            if not is_first_line_indented:
                # Flexible indentation matching
                match_result = PatchApplier._find_with_indentation_flexibility(new, search)
                if match_result is None:
                    raise PatchError(f"SEARCH chunk not found in source code. Search text: {repr(search[:100])}...")
                start_pos, end_pos, original_matched = match_result

                # If evolve regions are present, verify that the match lies wholly inside one evolve region
                if regions:
                    in_block = any(start <= start_pos < end for start, end in regions)
                    if not in_block:
                        raise PatchError(f"Attempted edit outside allowed evolve block. Edit position: {start_pos}, evolve regions: {regions}")

                # Preserve relative indentation structure in replacement
                replace = PatchApplier._preserve_relative_indentation(original_matched, replace)
                new = new[:start_pos] + replace + new[end_pos:]
            else:
                # Exact find/replace (no indentation logic)
                idx = new.find(search)
                if idx == -1:
                    raise PatchError(f"SEARCH chunk not found in source code. Search text: {repr(search[:100])}...")
                start_pos = idx
                end_pos = idx + len(search)

                # If evolve regions are present, verify that the match lies wholly inside one evolve region
                if regions:
                    in_block = any(start <= start_pos < end for start, end in regions)
                    if not in_block:
                        raise PatchError(f"Attempted edit outside allowed evolve block. Edit position: {start_pos}, evolve regions: {regions}")

                new = new[:start_pos] + replace + new[end_pos:]

        if new == source:
            raise PatchError("No changes were made to the source code")
        return new