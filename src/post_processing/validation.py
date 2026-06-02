"""HTML validation and comparison utilities."""

from ..html_utils import strip_auto_labels
from .html_operations import clean_html_formatting


def compare_html_allow_auto_labels(merged_html: str, original_html: str) -> bool:
    """
    Compare two HTML strings character-by-character, considering them equivalent
    if the only differences are the presence or placement of <auto_label ...>
    and </auto_label> tags, empty formatting tags, or redundant tag pairs.

    This function is used to verify that the merging process hasn't introduced
    unwanted changes to the original HTML structure.

    Args:
        merged_html: The merged HTML with auto_labels
        original_html: The original HTML without auto_labels

    Returns:
        bool: True if the HTMLs match after normalization, False otherwise
              Prints detailed diff information on mismatch
    """
    
    # Strip auto_labels and clean formatting artifacts
    a = strip_auto_labels(merged_html)
    b = strip_auto_labels(original_html)
    
    # Clean HTML formatting (removes empty tags and redundant pairs)
    a = clean_html_formatting(a)
    b = clean_html_formatting(b)
    
    if a == b:
        print("   ✓ HTMLs match after normalization (ignoring auto_label tags and formatting artifacts)")
        return True
    
    # Find first index of difference
    min_len = min(len(a), len(b))
    diff_idx = None
    for i in range(min_len):
        if a[i] != b[i]:
            diff_idx = i
            print(f"   ✗ Difference at index {diff_idx}: '{a[i]}' vs '{b[i]}'")
            break
    if diff_idx is None and len(a) != len(b):
        diff_idx = min_len
    
    # Print a small window around the difference
    if diff_idx is not None:
        start = max(0, diff_idx - 50)
        end_a = min(len(a), diff_idx + 50)
        end_b = min(len(b), diff_idx + 50)
        print("   ✗ Difference found (ignoring auto_label):")
        print("--- merged_html (stripped) ---")
        print(a[start:end_a])
        print("--- original_html (stripped) ---")
        print(b[start:end_b])
    else:
        print("   ✗ Difference detected but could not locate index")
    
    return False
