"""Main post-processing orchestration for document-level HTML processing."""

from ..tokenizer_utils import tokenize, decode
from ..html_utils import is_auto_label_tag, is_tag_token

from .token_operations import merge_tokens_general, flatten_token_chunks
from .validation import compare_html_allow_auto_labels
from .bracket_fixing import correct_tokens_brackets, check_tokens_brackets
from .html_operations import add_attributes_to_auto_labels, clean_html_formatting


def chunks_to_html(processed_chunks, html_content):
    """
    Main orchestration function for document-level post-processing.
    
    This function:
    1. Flattens processed chunks from the model into a single token list
    2. Tokenizes the original HTML content
    3. Merges original tokens with processed tokens, preserving auto_label insertions
    4. Validates the merge against the original HTML
    5. Corrects any bracket nesting issues introduced by merging
    6. Validates bracket coherence
    7. Cleans up HTML formatting
    8. Adds label scheme attributes to auto_labels
    
    Args:
        processed_chunks: List of lists of tokens from the model output
        html_content: Original HTML content string
    
    Returns:
        str: Processed HTML with properly annotated auto_label tags
    
    Raises:
        AssertionError: If validation or bracket checking fails
    """
    
    print("\n" + "="*80)
    print("DOCUMENT LEVEL POST-PROCESSING")
    print("="*80)
    
    # =====================================================================
    # Step 1: Flatten processed chunks
    # =====================================================================
    print("\n[Step 1] Flattening processed chunks...")
    processed_tokens_flat = flatten_token_chunks(processed_chunks)
    print(f"   ✓ Flattened to {len(processed_tokens_flat)} tokens")
    
    # =====================================================================
    # Step 2: Tokenize original HTML and merge with processed tokens
    # =====================================================================
    print("\n[Step 2] Tokenizing original HTML...")
    original_tokens = tokenize(html_content)
    print(f"   ✓ Tokenized to {len(original_tokens)} tokens")
    
    print("\n[Step 3] Merging original tokens with processed tokens...")
    print("   → Preserving original formatting while inserting auto_labels...")
    processed_html_content_tokens = merge_tokens_general(
        original_tokens=original_tokens,
        derived_tokens=processed_tokens_flat,
        is_protected_func=lambda tok: is_auto_label_tag(tok) != 0,
        is_opening_protected_func=lambda tok: is_auto_label_tag(tok) == 1,
        is_tag_token_func=lambda tok: is_tag_token(tok),
        log=False
    )
    print(f"   ✓ Merged to {len(processed_html_content_tokens)} tokens")
    
    # =====================================================================
    # Step 4: Validate merge against original HTML
    # =====================================================================
    print("\n[Step 4] Validating merge against original HTML...")
    comparison_result = compare_html_allow_auto_labels(
        decode(processed_html_content_tokens), 
        html_content
    )
    assert comparison_result, (
        "The processed HTML content does not match the original HTML content "
        "when ignoring auto_label tags. Please check the merging and "
        "post-processing steps for errors."
    )
    
    # =====================================================================
    # Step 5: Fix bracket nesting issues
    # =====================================================================
    print("\n[Step 5] Correcting bracket nesting issues...")
    processed_html_content_tokens_corrected = correct_tokens_brackets(processed_html_content_tokens)
    print(f"   ✓ Corrected {len(processed_html_content_tokens_corrected)} tokens")
    
    # =====================================================================
    # Step 6: Validate bracket coherence
    # =====================================================================
    print("\n[Step 6] Validating bracket coherence...")
    ok, message, position, context = check_tokens_brackets(processed_html_content_tokens_corrected)
    assert ok, (
        f"The brackets in the merged tokens are not balanced: {message} "
        f"(at position {position}). Please check the merging and bracket "
        f"correction steps for errors.\nContext: {context}"
    )
    print(f"   ✓ {message}")
    
    # =====================================================================
    # Step 7: Clean HTML formatting
    # =====================================================================
    print("\n[Step 7] Cleaning HTML formatting...")
    processed_html = decode(processed_html_content_tokens_corrected)
    processed_html_cleaned = clean_html_formatting(processed_html)
    print(f"   ✓ Cleaned HTML length: {len(processed_html_cleaned)} characters")
    
    # =====================================================================
    # Step 8: Add label scheme attributes
    # =====================================================================
    print("\n[Step 8] Adding label scheme attributes...")
    processed_html_content = add_attributes_to_auto_labels(processed_html_cleaned)
    print(f"   ✓ Added attributes to auto_label tags")
    
    # =====================================================================
    # Final result
    # =====================================================================
    print("\n" + "="*80)
    print("✓ POST-PROCESSING COMPLETE")
    print("="*80)
    print(f"Final HTML length: {len(processed_html_content)} characters\n")
    
    return processed_html_content
