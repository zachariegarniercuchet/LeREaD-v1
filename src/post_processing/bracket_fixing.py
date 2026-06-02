"""Bracket and formatting tag fixing utilities."""

from ..html_utils import is_auto_label_tag, is_tag_token, is_opening_tag, get_tag_name, is_closing_tag, is_fmt_tag


def correct_tokens_brackets(tokens, fmt_tags = {"i", "b", "strong", "u", "em", "mark", "span"}):
    """
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
    
    The function processes the following problems caused by auto_label insertion:
    CASE 1: ERROR DETECTED, and one of the tag is directly outside or inside auto_label without any space: it has to get in or get out
        <i><auto_label labelname="decision">Some text</i> Some text </auto_label> 
        -> <auto_label labelname="decision"><i>Some text</i> Some text </auto_label> 
        
        <i>Some text <auto_label labelname="decision"></i> Some text </auto_label> 
        -> <i>Some text </i><auto_label labelname="decision"> Some text </auto_label> 
        
        Some text <auto_label labelname="decision">Some text <i> Some text </auto_label></i> 
        -> Some text <auto_label labelname="decision">Some text <i> Some text </i></auto_label>
    
    CASE 2: ERROR DETECTED but no label is directly next to a autotag:
        <span class=""> Some text <auto_label labelname="decision">Some text </span> Some text </auto_label> 
        -> <span class=""> Some text </span><auto_label labelname="decision"><span class=""> Some text </span> Some text </auto_label>
        
        <auto_label labelname="decision">Some text <span class=""> Some text </auto_label> Some text </span> 
        -> <auto_label labelname="decision">Some text <span class=""> Some text </span></auto_label><span class="" Some text </span>
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
    
    Args:
        tokens: List of tokens to verify
        fmt_tags: Set of formatting tag names to check (default: common HTML formatting tags)
        ctx: Number of tokens to show in error context (default: 10)
    
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
