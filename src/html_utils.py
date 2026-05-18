from bs4 import BeautifulSoup
import itertools
import re
    
def extract_body(html_content: str) -> str:
    """
    Extract only the body content from HTML, excluding style, script, and head tags.
    Returns the exact string representation of the <body> element to keep reversibility.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    body = soup.find('body')
    if body is not None:
        return str(body)
    print("   ⚠ Warning: No <body> tag found, returning original content")
    return html_content

def remove_bookmarks(tokens: list) -> list:
    """Remove all 
      - <htmllabelizer_bookmark ...> tokens 
      - </htmllabelizer_bookmark ...> closing tokens 
      - tokens in between 
    from the token list."""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith('<htmllabelizer_bookmark'):
            # Find closing tag
            j = i + 1
            while j < len(tokens):
                if tokens[j].startswith('</htmllabelizer_bookmark'):
                    break
                j += 1
            # Remove from i to j (inclusive)
            del tokens[i:j+1]
            continue
        i += 1

    return [tok for tok in tokens if not tok.startswith('<htmllabelizer_bookmark')]

def is_tag_token(tok: str) -> bool:
    """Return True if token looks like an HTML tag (e.g., <...>)."""
    return len(tok) >= 3 and tok[0] == '<' and tok[-1] == '>'


def is_auto_label_tag(tok: str) -> bool:
    """Return 1 if token is an opening, 2 if it is a closing auto_label tag, 0 otherwise"""
    if not is_tag_token(tok):
        return 0
    # Accept variations with attributes on opening tag
    if tok.lower().startswith('<auto_label'):
        return 1
    if tok.lower().startswith('</auto_label'):
        return 2
    return 0

def is_manual_label_tag(tok):
    """Return 1 if token is an opening, 2 if it is a closing manual_label tag, 0 otherwise"""
    if not is_tag_token(tok):
        return 0
    # Accept variations with attributes on opening tag
    if tok.lower().startswith('<manual_label'):
        return 1
    if tok.lower().startswith('</manual_label'):
        return 2
    return 0

def strip_auto_labels(html: str) -> str:
    """Remove all <auto_label ...> and </auto_label> tags from html."""
    # Remove opening tags with any attributes
    html_no_open = re.sub(r"<\s*auto_label\b[^>]*>", "", html, flags=re.IGNORECASE)
    # Remove closing tags
    html_no_tags = re.sub(r"<\s*/\s*auto_label\s*>", "", html_no_open, flags=re.IGNORECASE)
    return html_no_tags




def is_fmt_tag(tok: str, fmt_tags: set) -> bool:
    """Check if token is a formatting tag (opening or closing)."""
    if not is_tag_token(tok):
        return False
    
    tag_name = get_tag_name(tok)
    return tag_name in fmt_tags

def is_opening_tag(tok: str) -> bool:
    """Check if token is an opening tag (not closing)."""
    return is_tag_token(tok) and not tok.startswith('</')


def is_closing_tag(tok: str) -> bool:
    """Check if token is a closing tag."""
    return is_tag_token(tok) and tok.startswith('</')

def get_tag_name(tok: str) -> str:
    """
    Extract the tag name from a token.
    Examples:
        '<i>' -> 'i'
        '</i>' -> 'i'
        '<span class="test">' -> 'span'
        '<auto_label labelname="decision">' -> 'auto_label'
    """
    if not is_tag_token(tok):
        return ""
    
    # Remove < and >
    content = tok[1:-1]
    
    # Remove leading / for closing tags
    if content.startswith('/'):
        content = content[1:]
    
    # Split on whitespace to get just the tag name (handles attributes)
    tag_name = content.split()[0] if content else ""
    
    return tag_name


def clean_html_formatting(html: str, tags_to_clean: set = None, debug: bool = False) -> str:
    """
    Clean HTML by removing useless formatting artifacts WITHOUT changing any text content.
    
    This function performs comprehensive HTML normalization by:
    1. PASS 1: Remove ALL empty tags (tags with NO children)
    2. PASS 2: Merge ALL adjacent identical tags (same name + attributes, no text between)
    3. Repeat until no more changes
    
    CRITICAL: Only merges tags that are truly adjacent with no text nodes between them.
    This preserves ALL text content (including spaces) for character-by-character comparison.
    
    Helps normalize HTML for comparison by removing artifacts like:
    - Empty tags: <i></i>, <span class="..."></span>
    - Adjacent empty + non-empty: <i></i><i>text</i> → <i>text</i>
    - Adjacent identical tags: <b>a</b><b>b</b> → <b>ab</b>
    
    Does NOT merge tags with ANY content between them:
    - <i>a</i> <i>b</i> → stays as is (space preserved)
    - <b>a</b>text<b>b</b> → stays as is
    
    Args:
        html: HTML string to clean
        tags_to_clean: Set of tag names to check. If None, checks common formatting tags.
        debug: If True, print debug information about cleaning operations
    
    Returns:
        Cleaned HTML string with useless formatting removed, all text preserved
    
    Examples:
        >>> clean_html_formatting('<b>text</b><b>more</b>')
        '<b>textmore</b>'
        >>> clean_html_formatting('<i></i><i>text</i>')
        '<i>text</i>'
        >>> clean_html_formatting('<span></span>text')
        'text'
        >>> clean_html_formatting('<i>a</i> <i>b</i>')  # space preserved
        '<i>a</i> <i>b</i>'
    """
    if tags_to_clean is None:
        tags_to_clean = {"span", "i", "b", "strong", "u", "em", "mark", "sup", "sub"}
    
    soup = BeautifulSoup(html, 'html.parser')
    
    max_iterations = 50  # Safety limit
    total_empty_removed = 0
    total_merged = 0
    
    if debug:
        print(f"\n=== Starting clean_html_formatting ===")
        print(f"Tags to clean: {tags_to_clean}")
        print(f"Input length: {len(html)} chars")
    
    # Loop until no more changes can be made
    for iteration in range(max_iterations):
        if debug:
            print(f"\n--- Iteration {iteration + 1} ---")
        
        # PASS 1: Remove ALL empty tags in one complete pass
        empty_removed_this_pass = 0
        for tag_name in tags_to_clean:
            while True:  # Keep removing until no more empty tags of this type
                tags = soup.find_all(tag_name)
                found_empty = False
                
                for tag in tags:
                    # Check if tag has NO children (not even text/whitespace)
                    if len(list(tag.children)) == 0:
                        if debug:
                            print(f"  [PASS 1] Removing empty <{tag_name}> tag")
                        tag.decompose()
                        empty_removed_this_pass += 1
                        found_empty = True
                        break  # Restart search after modification
                
                if not found_empty:
                    break  # No more empty tags of this type
        
        total_empty_removed += empty_removed_this_pass
        if debug and empty_removed_this_pass > 0:
            print(f"  [PASS 1] Removed {empty_removed_this_pass} empty tags")
        
        # PASS 2: Merge ALL adjacent identical tags in one complete pass
        merged_this_pass = 0
        for tag_name in tags_to_clean:
            while True:  # Keep merging until no more adjacent pairs of this type
                tags = soup.find_all(tag_name)
                found_merge = False
                
                for tag in tags:
                    # Look at the next sibling - must be immediate, no text between
                    next_sib = tag.next_sibling
                    
                    # Only merge if next sibling is same tag type with same attributes
                    if (next_sib and 
                        hasattr(next_sib, 'name') and 
                        next_sib.name == tag_name and
                        dict(tag.attrs) == dict(next_sib.attrs)):
                        
                        if debug:
                            tag_str = str(tag)[:60] + "..." if len(str(tag)) > 60 else str(tag)
                            next_str = str(next_sib)[:60] + "..." if len(str(next_sib)) > 60 else str(next_sib)
                            print(f"  [PASS 2] Merging <{tag_name}> tags:")
                            print(f"           First:  {tag_str}")
                            print(f"           Second: {next_str}")
                        
                        # Merge: move all contents from next_sib into tag
                        children_to_move = list(next_sib.children)
                        for child in children_to_move:
                            tag.append(child)
                        
                        # Remove the now-empty next tag
                        next_sib.decompose()
                        merged_this_pass += 1
                        found_merge = True
                        break  # Restart search after modification
                
                if not found_merge:
                    break  # No more adjacent pairs of this type
        
        total_merged += merged_this_pass
        if debug and merged_this_pass > 0:
            print(f"  [PASS 2] Merged {merged_this_pass} adjacent tag pairs")
        
        # If no changes in this iteration, we're done
        if empty_removed_this_pass == 0 and merged_this_pass == 0:
            if debug:
                print(f"\n=== Cleaning complete after {iteration + 1} iterations ===")
                print(f"Total empty tags removed: {total_empty_removed}")
                print(f"Total tag pairs merged: {total_merged}")
                print(f"Output length: {len(str(soup))} chars")
            break
    
    return str(soup)





