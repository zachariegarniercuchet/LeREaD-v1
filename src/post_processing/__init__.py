"""
Post-processing module for document-level HTML reconstruction and verification.

This module handles the reconstruction of HTML documents with auto-label annotations
by merging processed model output with original HTML structure while preserving
formatting and ensuring validity.

Main Functions:
    - main_post_processing: Orchestrates the entire post-processing pipeline

Format Conversion:
    - extract_start_end_tokens: Extract tokens between markers
    - simplified_to_normal_form: Convert simplified label format
    - apply_post_processing_transforms: Apply all format conversions

Token Operations:
    - merge_tokens_general: Merge original tokens with derived tokens
    - _tokens_equivalent: Check token equivalence

Bracket/Formatting:
    - correct_tokens_brackets: Fix formatting tag nesting
    - check_tokens_brackets: Validate bracket coherence

HTML Operations:
    - add_attributes_to_auto_labels: Add label scheme attributes

Validation:
    - compare_html_allow_auto_labels: Compare HTML ignoring auto_labels

Usage:
    from src.post_processing import main_post_processing
    
    result = main_post_processing(processed_chunks, original_html)
"""

from .main import chunks_to_html

# Import utilities for advanced usage
from .token_operations import merge_tokens_general
from .format_conversion import (
    extract_start_end_tokens,
    simplified_to_normal_form,
    apply_post_processing_transforms,
)
from .bracket_fixing import correct_tokens_brackets, check_tokens_brackets
from .html_operations import add_attributes_to_auto_labels
from .validation import compare_html_allow_auto_labels

__all__ = [
    # Main function
    "chunks_to_html",
    # Token operations
    "merge_tokens_general",
    # Format conversion
    "extract_start_end_tokens",
    "simplified_to_normal_form",
    "apply_post_processing_transforms",
    # Bracket fixing
    "correct_tokens_brackets",
    "check_tokens_brackets",
    # HTML operations
    "add_attributes_to_auto_labels",
    # Validation
    "compare_html_allow_auto_labels",
]
