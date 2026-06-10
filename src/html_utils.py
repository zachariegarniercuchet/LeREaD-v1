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

def is_specific_label_tag(tok, labelnames, labeltype='auto_label'):
    """Return 1 if token is an opening, 2 if it is a closing auto_label tag with the specified labelname, 0 otherwise"""
    if not is_tag_token(tok):
        return False
    # Check for opening tag with labelname attribute
    for labelname in labelnames:
        if tok.lower().startswith(f'<{labeltype}') and f'labelname="{labelname}"' in tok.lower():
            return True
    return False

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
        '<secondary sources labelname="decision">' -> 'secondary sources'
    """
    if not is_tag_token(tok):
        return ""
    
    # Remove < and >
    content = tok[1:-1].strip()
    
    # Remove leading / for closing tags
    if content.startswith('/'):
        content = content[1:]
    
    # Split on whitespace and accumulate words into the tag name
    # until we hit a word that contains '=' (attribute key=value)
    parts = content.split()
    name_parts = []
    for part in parts:
        if '=' in part:
            break
        name_parts.append(part)
    
    return " ".join(name_parts)










