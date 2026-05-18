import re
from .html_utils import is_auto_label_tag, is_manual_label_tag, is_tag_token, strip_auto_labels
from .tokenizer_utils import tokenize
from .htmlLabel import from_simplified
from .html_utils import clean_html_formatting, is_opening_tag, get_tag_name, is_closing_tag, is_fmt_tag


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


def _tokens_equivalent(tok1: str, tok2: str) -> bool:
        """
        Check if two tokens are equivalent.
        
        Tokens are equivalent if:
        1. They are exactly equal (tok1 == tok2)
        2. Both are pure whitespace (any combination of spaces, newlines, tabs, nbsp, etc.)
        3. Both are manual_label tags (opening or closing)
        
        Examples:
        - 'hello' == 'hello' -> True
        - 'hello' == 'world' -> False
        - ' ' == '\n\xa0\n\n\n\n' -> True (both are pure whitespace)
        - ',' == ',' -> True
        - ',' == ' ' -> False
        - '<manual_label labelname="mention">' == '<manual_label labelname="title">' -> True
        """
        # First check: exact equality
        if tok1 == tok2:
            return True
        
        # Second check: both are pure whitespace (different variants)
        if not tok1.strip() and not tok2.strip():
            return True
        
        # Third check: both are manual_label tags (opening or closing)
        
        if is_manual_label_tag(tok1) != 0 and is_manual_label_tag(tok2) != 0:
            return True
        
        # Otherwise, not equivalent
        return False


def merge_tokens_with_auto_labels(tokens: list[str], processed_tokens: list[str], output_dir=None, filename=None, log=False) -> list[str]:
    """
    Merge original tokens with processed tokens to produce the original text
    with only <auto_label ...> tags inserted from processed_tokens.

    Algorithm:
    - If tokens are equivalent (same or both whitespace): take original token, advance both indices
    - If tokens differ:
      - If processed token is a auto_label tag: it's an insertion, take it and advance idx2 only
      - Otherwise: take original token and advance idx1 only
    
    This handles insertions of <auto_label> tags in the processed version.
    """
    n1 = len(tokens)
    n2 = len(processed_tokens)
    result = []
    idx1 = 0
    idx2 = 0
    count = 0

    txt = ""

    while idx1 < n1 and idx2 < n2:
        t1 = tokens[idx1]
        t2 = processed_tokens[idx2]
        txt += f"TOKEN1 {t1}\n"
        txt += f"TOKEN2 {t2}\n"
        
        if _tokens_equivalent(t1, t2):
            # Tokens match: keep original and advance both
            result.append(t1)
            idx1 += 1
            idx2 += 1
            txt += f"COUNT {count}\n"
            
            count = 0
        else:
            count +=1
            # Tokens differ
            if is_auto_label_tag(t2) != 0: # meaning opening or closing auto tag

                if is_auto_label_tag(t2) == 1 : # Meaning opening auto tag
                    # Find the first non-tag word after the opening auto_label tag
                    next_idx = idx2 + 1
                    target_word = None
                    
                    # Look for the first non-tag token after the auto_label opening tag
                    while next_idx < n2:
                        next_token = processed_tokens[next_idx]
                        if not is_tag_token(next_token):
                            target_word = next_token
                            break
                        next_idx += 1
                    
                    if target_word is not None:
                        # Search forward in list1 to find the target word
                        search_idx = idx1
                        found = False
                        
                        while search_idx < n1:
                            if _tokens_equivalent(tokens[search_idx], target_word):
                                # Found the target word in list1
                                # Add all tokens from idx1 to search_idx (excluding search_idx)
                                while idx1 < search_idx:
                                    result.append(tokens[idx1])
                                    txt += f"PRE-TAG TOKEN: {tokens[idx1]}\n"
                                    idx1 += 1
                                
                                # Now add the opening auto_label tag
                                result.append(t2)
                                txt += f"OPENING AUTO_LABEL: {t2}\n"
                                idx2 += 1
                                
                                found = True
                                break
                            search_idx += 1
                        
                        if not found:
                            # If target word not found in list1, just add the auto_label tag
                            result.append(t2)
                            txt += f"AUTO_LABEL (no match found): {t2}\n"
                            idx2 += 1
                    else:
                        # No non-tag token found after auto_label, just add it
                        result.append(t2)
                        txt += f"AUTO_LABEL (no target): {t2}\n"
                        idx2 += 1
                        
                else : # Closing auto tag
                # t2 is a closing auto_label tag (insertion in processed version)
                    result.append(t2)
                    idx2 += 1
            else:
                # t2 is not a auto_label: keep original token
                result.append(t1)
                idx1 += 1
        txt += f"DECISION : {result[-1]}\n"

    # Append remaining tokens from original (if any)
    if idx1 < n1:
        txt += f"t1 {len(tokens[idx1:])}\n"
        result.extend(tokens[idx1:])
    
    # Append remaining tokens from processed (if any, likely closing tags)
    if idx2 < n2:
        txt += f"idx2 {idx2}\n"
        txt += f"{processed_tokens[idx2:]}\n"
        txt += f"t2 {len(processed_tokens[idx2:])}\n"
        result.extend(processed_tokens[idx2:])
    
    if output_dir and filename:
        with open(f"{output_dir}/debug_merge_{filename}.txt", "w", encoding="utf-8") as f:
            f.write(txt)

    if log:
        print(f"   ✓ Merged {len(tokens)} original and {len(processed_tokens)} processed into {len(result)} tokens")
    return result


def merge_tokens_general(original_tokens: list[str], 
                        derived_tokens: list[str], 
                        is_protected_func,
                        is_opening_protected_func=None,
                        is_tag_token_func=None,
                        log: bool = False) -> list[str]:
    """
    GENERALIZED VERSION: Merge original tokens with derived tokens.
    
    Goal: Produce the original text with protected tokens (e.g., <sep>, <auto_label>) 
    inserted from the derived version.
    
    Algorithm:
    - If tokens are equivalent (same or both whitespace): take original token, advance both
    - If tokens differ:
      - If derived token is a CLOSING protected tag: emit it directly, advance idx2 only
      - If derived token is an OPENING protected tag:
          * Look ahead in derived for first non-tag word after the opening tag
          * Find that word in original (searching forward from idx1)
          * Flush all original tokens up to (but not including) that word
          * Emit the opening tag, advance idx2 only
          * (Handles cases like <i> appearing in original before the target word)
      - If derived token is a non-protected mismatch:
          * Try merging consecutive original tokens to match derived token
          * If merged: emit originals, advance idx1 by merge_count, idx2 by 1
          * Otherwise: emit t1, advance idx1 only
    
    Args:
        original_tokens: Original token list (without protected tokens)
        derived_tokens: Derived token list (with protected tokens inserted)
        is_protected_func: Returns True if a token is protected (opening or closing)
        is_opening_protected_func: Returns True if a token is an OPENING protected tag.
                                   If None, all protected tokens are treated as atomic
                                   insertions (old behaviour — no lookahead).
        is_tag_token_func: Returns True if a token is any kind of tag (used to skip
                           non-content tokens while searching for the anchor word).
                           Only used when is_opening_protected_func is provided.
                           If None, defaults to: lambda t: t.startswith('<') and t.endswith('>')
        log: Print debug information
    
    Returns:
        Merged token list with original tokens + protected tokens from derived
    
    Example (with opening-tag lookahead):
        original = ['The', 'cat', 'is', '<i>', 'great', '</i>']
        derived  = ['The', 'cat', 'is', '<protected>', 'great', '</protected>']
        is_protected      = lambda t: t in ('<protected>', '</protected>')
        is_opening        = lambda t: t == '<protected>'
        -> ['The', 'cat', 'is', '<i>', '<protected>', 'great', '</protected>', '</i>']
    """
    n1 = len(original_tokens)
    n2 = len(derived_tokens)
    result = []
    idx1 = 0
    idx2 = 0

    # ------------------------------------------------------------------ helpers

    def tokens_equivalent(tok1: str, tok2: str) -> bool:
        if tok1 == tok2:
            return True
        if not tok1.strip() and not tok2.strip():
            return True
        return False

    def _is_tag(token: str) -> bool:
        """Default tag detector: any <…> token."""
        if is_tag_token_func is not None:
            return is_tag_token_func(token)
        return token.startswith('<') and token.endswith('>')

    def try_merge_original_to_match_derived(start_idx: int, target: str) -> int:
        """
        Try to merge consecutive original tokens to match the derived token.
        Returns the number of original tokens that combine to match target (0 = no match).
        """
        if start_idx >= n1:
            return 0
        accumulated = ""
        for i in range(start_idx, min(start_idx + 10, n1)):
            accumulated += original_tokens[i]
            if accumulated == target:
                return i - start_idx + 1
        return 0

    def handle_opening_protected_tag(t2: str) -> None:
        """
        Port of the opening-tag logic from merge_tokens_with_auto_labels.

        1. Scan forward in derived to find the first non-tag word after t2.
        2. Search forward in original to find that anchor word.
        3. Flush all original tokens up to (but not including) the anchor word.
        4. Emit the opening protected tag and advance idx2.
        """
        nonlocal idx1, idx2

        # Step 1 – find the anchor word in derived
        next_idx = idx2 + 1
        target_word = None
        while next_idx < n2:
            next_token = derived_tokens[next_idx]
            if not _is_tag(next_token):
                target_word = next_token
                break
            next_idx += 1

        if target_word is not None:
            # Step 2 – search for the anchor word in original
            search_idx = idx1
            found = False
            while search_idx < n1:
                if tokens_equivalent(original_tokens[search_idx], target_word):
                    # Step 3 – flush originals up to (not including) the anchor
                    while idx1 < search_idx:
                        result.append(original_tokens[idx1])
                        if log:
                            print(f"Pre-tag flush: '{original_tokens[idx1]}'")
                        idx1 += 1
                    # Step 4 – emit the opening tag
                    result.append(t2)
                    if log:
                        print(f"Opening protected (lookahead): '{t2}'")
                    idx2 += 1
                    found = True
                    break
                search_idx += 1

            if not found:
                # Anchor word not in original — emit tag as-is
                result.append(t2)
                if log:
                    print(f"Opening protected (no anchor found): '{t2}'")
                idx2 += 1
        else:
            # No non-tag word after the opening tag — emit as-is
            result.append(t2)
            if log:
                print(f"Opening protected (no target): '{t2}'")
            idx2 += 1

    # ---------------------------------------------------------------- main loop

    while idx1 < n1 and idx2 < n2:
        t1 = original_tokens[idx1]
        t2 = derived_tokens[idx2]

        if tokens_equivalent(t1, t2):
            result.append(t1)
            if log:
                print(f"Match: '{t1}' == '{t2}' -> '{t1}'")
            idx1 += 1
            idx2 += 1

        else:
            if is_protected_func(t2):
                # Protected token — distinguish opening from closing if possible
                if is_opening_protected_func is not None and is_opening_protected_func(t2):
                    handle_opening_protected_tag(t2)
                else:
                    # Closing tag (or undifferentiated protected token) — emit directly
                    result.append(t2)
                    if log:
                        print(f"Closing/atomic protected: '{t1}' vs '{t2}' -> '{t2}'")
                    idx2 += 1
            else:
                # Non-protected mismatch — try merging original tokens
                merge_count = try_merge_original_to_match_derived(idx1, t2)
                if merge_count > 0:
                    for i in range(merge_count):
                        result.append(original_tokens[idx1 + i])
                    if log:
                        merged = original_tokens[idx1:idx1 + merge_count]
                        print(f"Merged {merge_count} tokens: {merged} -> '{t2}'")
                    idx1 += merge_count
                    idx2 += 1
                else:
                    result.append(t1)
                    if log:
                        print(f"Diff: '{t1}' vs '{t2}' -> '{t1}'")
                    idx1 += 1

    # Append remaining original tokens
    if idx1 < n1:
        result.extend(original_tokens[idx1:])

    # Append remaining derived tokens (typically trailing protected tags)
    if idx2 < n2:
        result.extend(derived_tokens[idx2:])

    if log:
        print(f"   ✓ Merged {n1} original + {n2} derived → {len(result)} tokens")
        print(f"   ✓ Added {len(result) - n1} protected tokens")

    return result




def add_attributes_to_auto_labels(html_content: str, label_scheme_path: str = None) -> str:
    """
    Add parent, style, verified, and all label scheme attributes to auto_label tags in HTML content.
    
    The function loads the label scheme from a JSON file to determine:
    - Parent relationships based on nesting context (parent is the immediately enclosing auto_label)
    - Colors for each label (converted to background-color style)
    - All attributes defined in the label scheme for each label
    
    Attributes are initialized as follows:
    - String type: empty string "" (or default value from scheme)
    - Checkbox type: "false" (or default value from scheme)
    - Dropdown type: default value from the label scheme
    
    Also adds verified="false" to all auto_label tags.
    
    Args:
        html_content: HTML string with auto_label tags
        label_scheme_path: Path to label_scheme.json file. If None, uses default path.
    
    Returns:
        Modified HTML string with all attributes added
    """
    import re
    import json
    import os
    
    # Default path to label scheme
    if label_scheme_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        label_scheme_path = os.path.join(current_dir, '..', '..', 'ressources', 'label_scheme.json')
    
    # Load label scheme
    try:
        with open(label_scheme_path, 'r', encoding='utf-8') as f:
            label_scheme = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Label scheme file not found at {label_scheme_path}. Using auto_label tags without modification.")
        return html_content
    
    # Build style mapping and attributes mapping from label scheme
    style_map = {}
    attributes_map = {}
    
    # Helper function to determine text color based on background brightness
    def get_text_color(hex_color: str) -> str:
        """Determine if text should be black or white based on background color brightness."""
        # Remove # if present
        hex_color = hex_color.lstrip('#')
        # Convert to RGB
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        # Calculate relative luminance
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return 'black' if luminance > 0.5 else 'white'
    
    # Convert hex color to rgb format
    def hex_to_rgb(hex_color: str) -> str:
        """Convert hex color to rgb() format."""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgb({r}, {g}, {b})"
    
    # Helper to get attribute value based on type
    def get_attribute_value(attr_config: dict) -> str:
        """Get the initial value for an attribute based on its type."""
        attr_type = attr_config.get('type', 'string')
        if attr_type == 'string':
            return attr_config.get('default', '')
        elif attr_type == 'checkbox':
            default = attr_config.get('default', False)
            return 'true' if default else 'false'
        elif attr_type == 'dropdown':
            return attr_config.get('default', '')
        else:
            return ''
    
    # Process label scheme to build style and attributes mappings
    for parent_label, parent_data in label_scheme.items():
        # Set style for top-level label
        if 'color' in parent_data:
            bg_color = hex_to_rgb(parent_data['color'])
            text_color = get_text_color(parent_data['color'])
            style_map[parent_label] = f"background-color: {bg_color}; color: {text_color};"
        
        # Set attributes for top-level label
        if 'attributes' in parent_data:
            attributes_map[parent_label] = parent_data['attributes']
        
        # Process sublabels
        if 'sublabels' in parent_data:
            for sublabel, sublabel_data in parent_data['sublabels'].items():
                # Set style for sublabel
                if 'color' in sublabel_data:
                    bg_color = hex_to_rgb(sublabel_data['color'])
                    text_color = get_text_color(sublabel_data['color'])
                    style_map[sublabel] = f"background-color: {bg_color}; color: {text_color};"
                
                # Set attributes for sublabel
                if 'attributes' in sublabel_data:
                    attributes_map[sublabel] = sublabel_data['attributes']
    
    # Pattern to match auto_label tags (opening and closing)
    tag_pattern = r'<(/?)auto_label([^>]*)>'
    
    # Stack to track currently open auto_labels
    label_stack = []
    result_parts = []
    last_pos = 0
    
    for match in re.finditer(tag_pattern, html_content, flags=re.IGNORECASE):
        # Add text before this tag
        result_parts.append(html_content[last_pos:match.start()])
        
        is_closing = match.group(1) == '/'
        tag_content = match.group(2)
        
        if is_closing:
            # Closing tag - pop from stack
            if label_stack:
                label_stack.pop()
            result_parts.append(match.group(0))
        else:
            # Opening tag - extract labelname and determine parent
            labelname_match = re.search(r'labelname="([^"]*)"', tag_content)
            if labelname_match:
                labelname = labelname_match.group(1)
                
                # Parent is the labelname of the tag at the top of the stack (or "" if stack is empty)
                parent_value = label_stack[-1] if label_stack else ""
                parent_attr = f'parent="{parent_value}"'
                
                # Get style attribute from label scheme
                style = style_map.get(labelname, '')
                style_attr = f'style="{style}"' if style else ''
                
                # Add verified attribute
                verified_attr = 'verified="false"'
                
                # Get all attributes for this label from label scheme
                label_attrs = attributes_map.get(labelname, {})
                scheme_attrs = []
                for attr_name, attr_config in label_attrs.items():
                    attr_value = get_attribute_value(attr_config)
                    scheme_attrs.append(f'{attr_name}="{attr_value}"')
                
                # Remove existing parent/style/verified/scheme attributes if present
                tag_content = re.sub(r'\s*parent="[^"]*"', '', tag_content)
                tag_content = re.sub(r'\s*style="[^"]*"', '', tag_content)
                tag_content = re.sub(r'\s*verified="[^"]*"', '', tag_content)
                # Remove existing scheme attributes
                for attr_name in label_attrs.keys():
                    tag_content = re.sub(rf'\s*{re.escape(attr_name)}="[^"]*"', '', tag_content)
                
                # Build new tag with all attributes
                all_attrs = [parent_attr, style_attr, verified_attr] + scheme_attrs
                attrs_str = ' '.join(filter(None, all_attrs))  # Filter out empty strings
                new_tag = f'<auto_label{tag_content} {attrs_str}>'
                result_parts.append(new_tag)
                
                # Push this label onto the stack
                label_stack.append(labelname)
            else:
                # No labelname found, keep tag as-is
                result_parts.append(match.group(0))
        
        last_pos = match.end()
    
    # Add remaining text after last tag
    result_parts.append(html_content[last_pos:])
    
    return ''.join(result_parts)


def correct_tokens_brackets(tokens, fmt_tags = {"i", "b", "strong", "u", "em", "mark", "span"}):
    """"
    Fix formatting tag nesting issues caused by auto_label insertion.
    
    The function ensures that formatting tags don't cross auto_label boundaries.
    When a closing </auto_label> is encountered, all open formatting tags are closed
    before it, then reopened after it.
    
    CASE 1: Direct nesting issues
        <i><auto_label>text</i> more</auto_label> 
        -> <auto_label><i>text</i> more</auto_label>
    
    CASE 2: Indirect nesting issues
        <span>text <auto_label>more</span> text</auto_label>
        -> <span>text</span><auto_label><span>more</span> text</auto_label>


    The function process the folowing problem caused by auto_label insertion:
    CASE 1 : ERROR DETECTED, and one of the tag is directly outside or inside auto_label whitout anyspace : it as to get in or get out
        <i><auto_label labelname="decision">Some text</i> Some text </auto_label> -> <auto_label labelname="decision"><i>Some text</i> Some text </auto_label> 
        <i>Some text <auto_label labelname="decision"></i> Some text </auto_label> -> <i>Some text </i><auto_label labelname="decision"> Some text </auto_label> 
        Some text <auto_label labelname="decision">Some text <i> Some text </auto_label></i> -> Some text <auto_label labelname="decision">Some text <i> Some text </i></auto_label> 

    CASE 2 : ERROR DETECTED but no label is directly next to a autotag :
         <span class=""> Some text <auto_label labelname="decision">Some text </span> Some text </auto_label> -> <span class=""> Some text </span><auto_label labelname="decision"><span class=""> Some text </span> Some text </auto_label>
         <auto_label labelname="decision">Some text <span class=""> Some text </auto_label> Some text </span> -> <auto_label labelname="decision">Some text <span class=""> Some text </span></auto_label><span class="" Some text </span>
    """

    corrected = []
    fmt_stack = []  # Stack of (tag_name, full_opening_token)

    for tok in tokens:
        
        # Handle auto_label opening
        auto_label_type = is_auto_label_tag(tok)
        if auto_label_type == 1:  # Opening <auto_label>
            # Close all open formatting tags BEFORE opening auto_label
            to_reopen = []
            
            while fmt_stack:
                tag_name, open_tok = fmt_stack.pop()
                corrected.append(f"</{tag_name}>")
                to_reopen.append((tag_name, open_tok))
            
            # Now add the opening auto_label
            corrected.append(tok)
            
            # Reopen formatting tags INSIDE auto_label
            for tag_name, open_tok in reversed(to_reopen):
                corrected.append(open_tok)
                fmt_stack.append((tag_name, open_tok))
            
            continue
        
        # Handle auto_label closing
        if auto_label_type == 2:  # Closing </auto_label>
            # Close all open formatting tags BEFORE closing auto_label
            to_reopen = []
            
            while fmt_stack:
                tag_name, open_tok = fmt_stack.pop()
                corrected.append(f"</{tag_name}>")
                to_reopen.append((tag_name, open_tok))
            
            # Now add the closing auto_label
            corrected.append(tok)
            
            # Reopen formatting tags AFTER auto_label
            for tag_name, open_tok in reversed(to_reopen):
                corrected.append(open_tok)
                fmt_stack.append((tag_name, open_tok))
            
            continue
        
        # Handle opening formatting tags
        if is_opening_tag(tok):
            tag_name = get_tag_name(tok)
            if tag_name in fmt_tags:
                fmt_stack.append((tag_name, tok))
                corrected.append(tok)
                continue
        
        # Handle closing formatting tags
        if is_closing_tag(tok):
            tag_name = get_tag_name(tok)
            if tag_name in fmt_tags:
                # Remove matching opening tag from stack (search from end)
                for i in range(len(fmt_stack) - 1, -1, -1):
                    if fmt_stack[i][0] == tag_name:
                        fmt_stack.pop(i)
                        break
                
                corrected.append(tok)
                continue
        
        # All other tokens (text, non-fmt tags, etc.)
        corrected.append(tok)

    return corrected


def check_tokens_brackets(tokens, fmt_tags={"i", "b", "strong", "u", "em", "mark", "span"}, ctx=10):
    """
    Verify that tags are properly nested and matched.
    
    Returns:
        tuple: (ok, message, position, context)
            - ok (bool): True if brackets are coherent
            - message (str): Description of the result or error
            - position (int or None): Index of error token if any
            - context (str or None): Surrounding tokens for debugging
    """
    stack = []  # Stack of (tag_name, index, full_token)

    def context_at(i):
        """Get surrounding context for error reporting."""
        start = max(0, i - ctx)
        end = min(len(tokens), i + ctx + 1)
        snippet = tokens[start:end]
        return " ".join(snippet)

    for idx, tok in enumerate(tokens):
        
        # Skip non-tag tokens
        if not is_tag_token(tok):
            continue
        
        # Handle auto_label opening
        auto_label_type = is_auto_label_tag(tok)
        if auto_label_type == 1:  # Opening auto_label
            stack.append(("auto_label", idx, tok))
            continue
        
        # Handle auto_label closing
        if auto_label_type == 2:  # Closing auto_label
            if not stack:
                return False, "Closing </auto_label> with empty stack", idx, context_at(idx)
            
            open_name, open_idx, open_tok = stack.pop()
            
            if open_name != "auto_label":
                msg = f"Mismatched tags: opened {open_tok} at {open_idx}, closed </auto_label>"
                return False, msg, idx, context_at(idx)
            
            continue
        
        # Handle opening formatting tags
        if is_opening_tag(tok) and is_fmt_tag(tok, fmt_tags):
            tag_name = get_tag_name(tok)
            stack.append((tag_name, idx, tok))
            continue
        
        # Handle closing formatting tags
        if is_closing_tag(tok) and is_fmt_tag(tok, fmt_tags):
            tag_name = get_tag_name(tok)
            
            if not stack:
                return False, f"Closing tag {tok} with empty stack", idx, context_at(idx)
            
            open_name, open_idx, open_tok = stack.pop()
            
            if open_name != tag_name:
                msg = f"Mismatched tags: opened {open_tok} at {open_idx}, closed {tok}"
                return False, msg, idx, context_at(idx)
            
            continue

    # Check for unclosed tags
    if stack:
        name, idx, tok = stack[-1]
        return False, f"Unclosed tag {tok}", idx, context_at(idx)

    return True, "Brackets are coherent", None, None



def compare_html_allow_auto_labels(merged_html: str, original_html: str) -> bool:
    """
    Compare two HTML strings character-by-character, considering them equivalent
    if the only differences are the presence or placement of <auto_label ...>
    and </auto_label> tags, empty formatting tags, or redundant tag pairs.

    Returns True when equal after normalization, else prints
    a concise diff context and returns False.
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