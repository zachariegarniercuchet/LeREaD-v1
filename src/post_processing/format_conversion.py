"""Format conversion utilities for processing LLM output."""

import re

from ..htmlLabel import simplified_to_normal_form
from ..tokenizer_utils import tokenize


def extract_start_end_tokens(tokens: list) -> list:
    """
    Extract tokens between <start> and <end> markers.
    
    Args:
        tokens: List of tokens containing <start> and <end> markers
    
    Returns:
        List of tokens between markers, or original tokens if markers not found
    
    Raises:
        ValueError: If <start> found but <end> not found, or vice versa
    """
    try:
        start_index = tokens.index("<start>")
    except ValueError:
        print("   ⚠ Warning: <start> marker not found, returning original tokens")
        return tokens
    
    try:
        end_index = tokens.index("<end>")
    except ValueError:
        raise ValueError("<start> marker found but <end> marker missing")
    
    if end_index <= start_index:
        raise ValueError("<end> marker appears before <start> marker")
    
    extracted = tokens[start_index + 1 : end_index]
    print(f"   ✓ Extracted {len(extracted)} tokens between <start> and <end>")
    return extracted



def apply_post_processing_transforms(raw_output: str, use_simplified: bool = False, label_type: str = 'auto_label') -> list:
    """
    Apply all post-processing transformations to raw LLM output.
    
    Pipeline:
    1. Tokenize raw output
    2. Extract tokens between <start> and <end> markers
    3. Convert simplified format to normal form (if applicable)
    
    Args:
        raw_output: Raw text output from LLM
        use_simplified: Whether the LLM output uses simplified format
        label_type: Target label type ('auto_label' or 'manual_label')
    
    Returns:
        List of processed tokens ready for verification
    """
    # Step 1: Tokenize
    tokens = tokenize(raw_output)
    print(f"   → Step 1: Tokenized into {len(tokens)} tokens")
    
    # Step 2: Extract between <start> and <end>
    try:
        tokens = extract_start_end_tokens(tokens)
        print(f"   → Step 2: Extracted {len(tokens)} tokens between markers")
    except ValueError as e:
        print(f"   ⚠ Warning: {str(e)}")
    
    # Step 3: Convert simplified to normal form if needed
    if use_simplified:
        tokens = simplified_to_normal_form(tokens, label_type=label_type)
        print(f"   → Step 3: Converted to {label_type} format")
    
    return tokens
