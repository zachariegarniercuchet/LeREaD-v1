"""Format conversion utilities for processing LLM output."""

import re

from ..html_utils import is_tag_token
from ..tokenizer_utils import tokenize
from ..htmlLabel import from_simplified


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


def simplified_to_normal_form(tokens: list, label_type: str = 'auto_label') -> list:
    """
    Convert simplified label format to normal auto_label or manual_label format.
    
    Transforms:
        <decision> → <auto_label labelname="decision">
        </decision> → </auto_label>
        <title titletype="main"> → <auto_label labelname="title" titletype="main">
        </title> → </auto_label>
    
    Args:
        tokens: List of tokens potentially containing simplified label tags
        label_type: Either 'auto_label' or 'manual_label' (default: 'auto_label')
    
    Returns:
        List of tokens with normalized label format
    """
    if label_type not in ['auto_label', 'manual_label']:
        raise ValueError(f"label_type must be 'auto_label' or 'manual_label', got: {label_type}")
    
    normalized_tokens = []
    
    for token in tokens:
        # Check for opening tag: <...> but not </...> or <manual_label...> or <auto_label...>
        is_open = bool(re.fullmatch(r'<(?!\/|manual_label|auto_label)[^>]+>', token))
        
        # Check for closing tag: </...> but not </manual_label...> or </auto_label...>
        is_close = bool(re.fullmatch(r'<\/((?!manual_label|auto_label)[^>]+)>', token))
        
        if is_open:
            # Convert simplified opening tag to normal form
            html_label = from_simplified(token, label_type=label_type)
            normalized_tokens.append(html_label._token)
        elif is_close:
            # Convert simplified closing tag to normal form
            normalized_tokens.append(f'</{label_type}>')
        else:
            # Keep token as-is
            normalized_tokens.append(token)
    
    print(f"   ✓ Converted {len(tokens)} tokens from simplified to {label_type} format")
    return normalized_tokens


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
