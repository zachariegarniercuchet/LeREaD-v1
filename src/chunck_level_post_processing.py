"""
Post-processing utilities for LLM-generated annotated text.
Includes both transformation functions and merging utilities.
"""
import re
import json
import os
from typing import Optional, Set
from .htmlLabel import from_simplified
from .tokenizer_utils import tokenize

def load_label_scheme_names(scheme_path: Optional[str] = None) -> Set[str]:
    """
    Load all valid label names from the label scheme JSON file.
    
    Args:
        scheme_path: Path to label_scheme.json. If None, uses default location.
    
    Returns:
        Set of valid label names (lowercase) including main labels and sublabels
    """
    if scheme_path is None:
        # Default path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        scheme_path = os.path.join(current_dir, '..', '..', 'ressources', 'label_scheme.json')
    
    try:
        with open(scheme_path, 'r', encoding='utf-8') as f:
            scheme = json.load(f)
        
        label_names = set()
        
        # Add main labels
        for main_label in scheme.keys():
            label_names.add(main_label.lower())
            
            # Add sublabels if they exist
            if 'sublabels' in scheme[main_label]:
                for sublabel in scheme[main_label]['sublabels'].keys():
                    label_names.add(sublabel.lower())
        
        return label_names
    
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"   ⚠ Warning: Could not load label scheme from {scheme_path}: {e}")
        # Return default set if file can't be loaded
        return {'legislation', 'decision', 'secondary sources', 'title', 'citation', 
                'fragment', 'authors', 'source', 'unable to classify'}


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
    #print(f"   ✓ Extracted {len(extracted)} tokens between <start> and <end>")
    return extracted


def simplified_to_normal_form(tokens: list, label_type: str = 'auto_label', 
                             allowed_labels: Optional[Set[str]] = None) -> list:
    """
    Convert simplified label format to normal auto_label or manual_label format.
    Only converts tags that are in the label scheme, ignoring HTML formatting tags.
    
    Transforms:
        <decision> → <auto_label labelname="decision">
        </decision> → </auto_label>
        <title titletype="main"> → <auto_label labelname="title" titletype="main">
        </title> → </auto_label>
    
    Ignores HTML formatting tags like <i>, <b>, <span>, etc.
    
    Args:
        tokens: List of tokens potentially containing simplified label tags
        label_type: Either 'auto_label' or 'manual_label' (default: 'auto_label')
        allowed_labels: Set of allowed label names. If None, loads from label_scheme.json
    
    Returns:
        List of tokens with normalized label format
    """
    if label_type not in ['auto_label', 'manual_label']:
        raise ValueError(f"label_type must be 'auto_label' or 'manual_label', got: {label_type}")
    
    # Load label scheme if not provided
    if allowed_labels is None:
        allowed_labels = load_label_scheme_names()
    
    normalized_tokens = []
    
    for token in tokens:
        # Check for opening tag: <...> but not </...> or <manual_label...> or <auto_label...>
        is_open = bool(re.fullmatch(r'<(?!\/|manual_label|auto_label)[^>]+>', token))
        
        # Check for closing tag: </...> but not </manual_label...> or </auto_label...>
        is_close_match = re.fullmatch(r'<\/((?!manual_label|auto_label)[^>]+)>', token)
        
        if is_open:
            # Extract the tag name to check if it's a valid label
            # Tag name can contain spaces (e.g., "secondary sources")
            # Attributes always have '=' so we use that to differentiate tag name from attributes
            # Pattern: <tagname> or <tagname attr1=...> where tagname may contain spaces
            tag_match = re.match(r'<\s*([^>=]+?)(?:\s+\w+\s*=|\s*>)', token)
            if tag_match:
                tag_name = tag_match.group(1).strip().lower()
                
                # Only convert if it's in the label scheme
                if tag_name in allowed_labels:
                    html_label = from_simplified(token, label_type=label_type)
                    normalized_tokens.append(html_label._token)
                else:
                    # Keep HTML formatting tags as-is
                    normalized_tokens.append(token)
            else:
                normalized_tokens.append(token)
                
        elif is_close_match:
            # Extract the tag name from closing tag and strip whitespace
            tag_name = is_close_match.group(1).strip().lower()
            
            # Only convert if it's in the label scheme
            if tag_name in allowed_labels:
                normalized_tokens.append(f'</{label_type}>')
            else:
                # Keep HTML formatting closing tags as-is
                normalized_tokens.append(token)
        else:
            # Keep token as-is
            normalized_tokens.append(token)
    
    #print(f"   ✓ Converted {len(tokens)} tokens from simplified to {label_type} format")
    return normalized_tokens


def apply_post_processing_transforms(raw_output: str, use_simplified: bool = False, label_type: str = 'auto_label', cot: bool = False) -> list:
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
        cot: Whether the LLM output includes chain-of-thought reasoning
    Returns:
        List of processed tokens ready for verification
    """
    # Step 1: Tokenize
    tokens = tokenize(raw_output)
    #print(f"   → Step 1: Tokenized into {len(tokens)} tokens")
    
    # Step 2: Extract between <start> and <end>
    if cot:
        try:
            tokens = extract_start_end_tokens(tokens)
            #print(f"   → Step 2: Extracted {len(tokens)} tokens between markers")
        except ValueError as e:
            print(f"   ⚠ Warning: {str(e)}")
    
    # Step 3: Convert simplified to normal form if needed
    if use_simplified:
        tokens = simplified_to_normal_form(tokens, label_type=label_type)
        #print(f"   → Step 3: Converted to {label_type} format")
    
    return tokens