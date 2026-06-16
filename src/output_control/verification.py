"""
Verification utilities for validating LLM-generated annotated text.
These functions check the validity of processed chunks without modifying them.
"""

import sys
from pathlib import Path

from configs.config import LABEL_SCHEME, STRUCTURAL_LABELS

from src.html_utils import is_auto_label_tag
from src.htmlLabel import HTMLLabel

import json
from typing import Dict, List, Tuple, Optional
from bs4 import BeautifulSoup


# Label scheme definition based on label schemes.html



class VerificationResult:
    """Container for verification results with details about failures."""
    
    def __init__(self, passed: bool, error_type: str = None, details: str = None, tokens: list = None):
        self.passed = passed
        self.error_type = error_type  # "hallucination", "consistency", "label_scheme"
        self.details = details
        self.tokens = tokens  # The problematic tokens for debugging
    
    def __bool__(self):
        return self.passed
    
    def __repr__(self):
        if self.passed:
            return "VerificationResult(passed=True)"
        return f"VerificationResult(passed=False, error='{self.error_type}', details='{self.details}')"
    


def _find_token_differences(tokens1: list, tokens2: list, context: int = 5) -> str:
    """
    Find where two token lists differ and return a description.
    
    Args:
        tokens1: First token list
        tokens2: Second token list
        context: Number of tokens to show before/after difference
    
    Returns:
        String describing the difference location
    """
    min_len = min(len(tokens1), len(tokens2))
    
    # Find first difference
    for i in range(min_len):
        if tokens1[i] != tokens2[i]:
            start = max(0, i - context)
            end = min(min_len, i + context + 1)
            
            context1 = tokens1[start:end]
            context2 = tokens2[start:end]
            
            return (f"First difference at position {i}:\n"
                   f"  Original: ...{' '.join(context1)}...\n"
                   f"  Processed: ...{' '.join(context2)}...")
    
    # Lists match up to min_len but have different lengths
    if len(tokens1) != len(tokens2):
        return (f"Token lists have different lengths: "
               f"original={len(tokens1)}, processed={len(tokens2)}")
    
    return "No differences found"


def check_hallucination(original_tokens: list, processed_tokens: list) -> VerificationResult:
    """
    Verify that processed tokens match original tokens when labels are stripped.
    
    This checks if the LLM hallucinated or modified the original text content
    by comparing cleaned versions of both token lists.
    
    Args:
        original_tokens: Original token list
        processed_tokens: Processed token list with auto_label tags
        normalize: Whether to normalize whitespace during comparison
        keep_manual_label: Whether to keep manual_label tags during comparison
    
    Returns:
        VerificationResult with passed=True if no hallucination detected
    """
    from ..transforme_utils import clean_tokens
    
    original_cleaned = clean_tokens(
        original_tokens, 
        keep_manual_label=True,
        keep_auto_label=False,
        keep_bookmarks=False
    )
    
    processed_cleaned = clean_tokens(
        processed_tokens,
        keep_manual_label=True,
        keep_auto_label=False,
        keep_bookmarks=False
    )
    
    if original_cleaned == processed_cleaned:
        return VerificationResult(passed=True)
    
    # Find where they differ
    diff_details = _find_token_differences(original_cleaned, processed_cleaned)
    
    return VerificationResult(
        passed=False,
        error_type="hallucination",
        details=diff_details,
        tokens=processed_tokens
    )


def check_consistency(tokens: list) -> VerificationResult:
    """
    Verify that all auto_label opening and closing tags are properly balanced.
    
    Checks:
    - Every <auto_label ...> has a matching </auto_label>
    - Tags are properly nested (no cross-nesting)
    
    Args:
        tokens: List of tokens to check
    
    Returns:
        VerificationResult with passed=True if tags are balanced
    """
    
    
    
    stack = []
    
    for i, token in enumerate(tokens):
        tag_type = is_auto_label_tag(token)
        
        if tag_type == 1:  # Opening tag
            stack.append((token, i))
        elif tag_type == 2:  # Closing tag
            if not stack:
                return VerificationResult(
                    passed=False,
                    error_type="consistency",
                    details=f"Closing tag at position {i} without matching opening tag: {token}",
                    tokens=tokens
                )
            stack.pop()
    
    if stack:
        unclosed = [f"{tok} at position {pos}" for tok, pos in stack]
        return VerificationResult(
            passed=False,
            error_type="consistency",
            details=f"Unclosed tags: {', '.join(unclosed)}",
            tokens=tokens
        )
    
    return VerificationResult(passed=True)


def check_label_scheme(tokens: list, allowed_labels: Optional[List[str]] = None) -> VerificationResult:
    """
    Verify that all label names and attributes conform to the label scheme.
    
    Checks:
    - Label names are in the allowed set (LABEL_SCHEME or custom allowed_labels)
    - Attributes match the expected attributes for each label type
    - No invalid or random label names
    
    Args:
        tokens: List of tokens to check
        allowed_labels: Optional list of allowed label names. If None, uses LABEL_SCHEME keys.
    
    Returns:
        VerificationResult with passed=True if all labels conform to scheme
    """

    
    # Determine allowed label names
    if allowed_labels is None:
        allowed_labels = list(LABEL_SCHEME.keys())
    
    # Normalize allowed labels to lowercase for case-insensitive comparison
    allowed_labels_lower = [label.lower() for label in allowed_labels]
    
    invalid_labels = []
    invalid_attributes = []
    
    for i, token in enumerate(tokens):
        if is_auto_label_tag(token) == 1:  # Opening tag only
            try:
                label = HTMLLabel(token)
                label_name = label.name.lower()
                
                
                # Check if label name is allowed
                if label_name not in allowed_labels_lower:
                    invalid_labels.append(f"'{label.name}' at position {i}")
                    continue
                
                # Check attributes if label is in LABEL_SCHEME
                if label_name in LABEL_SCHEME:
                    expected_attrs = LABEL_SCHEME[label_name]["attributes"]
                    actual_attrs = [k for k in label.attributes.keys() if k != "labelname"]
                    
                    # Find unexpected attributes
                    unexpected = [attr for attr in actual_attrs if attr not in expected_attrs]
                    if unexpected:
                        invalid_attributes.append(
                            f"Label '{label.name}' at position {i} has unexpected attributes: {unexpected}"
                        )
            
            except ValueError as e:
                # Couldn't parse the label
                invalid_labels.append(f"Unparseable label at position {i}: {str(e)}")
    
    # Build result
    if invalid_labels or invalid_attributes:
        details_parts = []
        if invalid_labels:
            details_parts.append(f"Invalid label names: {', '.join(invalid_labels)}")
        if invalid_attributes:
            details_parts.append(f"Invalid attributes: {'; '.join(invalid_attributes)}")
        
        return VerificationResult(
            passed=False,
            error_type="label_scheme",
            details=" | ".join(details_parts),
            tokens=tokens
        )
    
    return VerificationResult(passed=True)


def check_nesting(tokens: list) -> VerificationResult:
    """
    Verify that structural auto_label tags are properly nested inside a valid
    parent auto_label tag.

    Checks:
    - 'title', 'citation', 'source', 'authors', 'fragment' must have at least one
      auto_label ancestor (i.e. must not appear at the top level)
    - 'legislation', 'decision', 'secondary sources' are always valid
      regardless of nesting

    Args:
        tokens: List of tokens to check

    Returns:
        VerificationResult with passed=True if all nesting rules are respected
    """
    

    # Stack holds the labelname of every currently-open auto_label tag
    stack: list[str] = []
    nesting_errors: list[str] = []

    for i, token in enumerate(tokens):
        tag_type = is_auto_label_tag(token)

        if tag_type == 1:  # Opening tag
            try:
                label = HTMLLabel(token)
                label_name = label.name.lower()
            except ValueError as e:
                # Unparseable tags are the concern of check_label_scheme, skip here
                stack.append("__unknown__")
                continue

            # Check nesting rule for structural labels
            if label_name in STRUCTURAL_LABELS and not stack:
                nesting_errors.append(
                    f"'{label.name}' at position {i} appears at top level "
                    f"(must be nested inside a parent auto_label)"
                )

            stack.append(label_name)

        elif tag_type == 2:  # Closing tag
            if stack:
                stack.pop()

    if nesting_errors:
        return VerificationResult(
            passed=False,
            error_type="nesting",
            details=f"Nesting violations: {' | '.join(nesting_errors)}",
            tokens=tokens,
        )

    return VerificationResult(passed=True)


def verify_processed_chunk(original_tokens: list, processed_tokens: list,
                          allowed_labels: Optional[List[str]] = None) -> VerificationResult:
    """
    Run all verification checks on a processed chunk.
    
    Performs four checks in order:
    1. Hallucination check (text content unchanged)
    2. Consistency check (tags properly balanced)
    3. Label scheme check (labels and attributes valid)
    4. Nesting check (structural labels properly nested)
    
    Returns first failure encountered, or success if all pass.
    
    Args:
        original_tokens: Original token list before processing
        processed_tokens: Processed token list with auto_label tags
        allowed_labels: Optional list of allowed label names
        check_scheme: Whether to check label scheme compliance (default: True)
    
    Returns:
        VerificationResult with details about first failure, or success
    """
    # Check 1: Hallucination
    result = check_hallucination(original_tokens, processed_tokens)
    if not result:
        return result
    
    # Check 2: Consistency
    result = check_consistency(processed_tokens)
    if not result:
        return result
    
    # Check 3: Label scheme
    result = check_label_scheme(processed_tokens, allowed_labels)
    if not result:
        return result
        
    result = check_nesting(processed_tokens)
    if not result:
        return result
    
    return VerificationResult(passed=True)



