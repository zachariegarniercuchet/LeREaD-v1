import re 
from src.tokenizer_utils import tokenize
from bs4 import BeautifulSoup

class ReferenceMention:

    def __init__(self, html_str: str):
        """
        html_str for example: <manual_label docid="Laicity Act" labelname="legislation" parent="" style="background-color: rgb(118, 206, 222); color: black;" 
        uri="https://canlii.ca/t/56lzp" verified="false">
        <manual_label labelname="title" parent="legislation" style="background-color: rgb(147, 196, 125); color: black;" titletype="alias" 
        verified="false">Laicity\nAct</manual_label></manual_label>
        """
        

        self.soup = BeautifulSoup(html_str, "html.parser")

        root = self.soup.find(["manual_label", "auto_label"])
        self.html_tag = HTMLLabel(str(root).split(">")[0] + ">")

        self.text = self.get_text()

        self.name = self.html_tag.name

        self.sublabels = self.get_sublabels()

    @property
    def html_str(self):
        # Synchronize the root tag with html_tag
        root = self.soup.find(["manual_label", "auto_label"])

        # Replace tag attributes
        root.attrs.clear()
        root.attrs.update(self.html_tag.attributes)

        return str(self.soup)


    def get_text(self) -> str:
        """
        Extract the text content from the HTML label, excluding the opening and closing tags.
        """
        return self.soup.get_text(strip=True)  # Use BeautifulSoup to get clean text content
    
    def get_sublabels(self) -> list[str]:
        
        children = self.soup.find_all(
            ["manual_label", "auto_label"],
            attrs={"parent": lambda v: v is not None and v != ""}
        )

        children_name = [child.get("labelname", "")
                    for child in children]
        return children_name
    
    def __str__(self):
        return str(self.html_str)

class HTMLLabel:
    """
    Parse and represent manual_label or auto_label HTML tokens.
    
    Example tokens:
        <manual_label labelname="Authority_Mention" jurisdiction="Canada">
        <auto_label labelname="Legal_Issue" confidence="0.95">
    """
    
    def __init__(self, token: str):
        """
        Initialize HTMLLabel from a token string.
        
        Args:
            token: String token like '<manual_label labelname="xyz" attr="val">'
        
        Raises:
            ValueError: If token is not a valid manual_label or auto_label tag
        """
        if not self._is_valid_label_token(token):
            raise ValueError(f"Token is not a valid manual_label or auto_label: {token}")
        
        self._token = token
        self._label_type = self._detect_label_type(token)
        self._attributes = self._parse_attributes(token)
        
        if 'labelname' not in self._attributes:
            raise ValueError(f"Token missing 'labelname' attribute: {token}")
    
    def _is_valid_label_token(self, token: str) -> bool:
        """Check if token is a valid manual_label or auto_label opening tag."""
        if not token.startswith('<') or not token.endswith('>'):
            return False
        lower = token.lower()
        return lower.startswith('<manual_label') or lower.startswith('<auto_label')
    
    def _detect_label_type(self, token: str) -> str:
        """Detect whether token is 'manual_label' or 'auto_label'."""
        if token.lower().startswith('<manual_label'):
            return 'manual_label'
        elif token.lower().startswith('<auto_label'):
            return 'auto_label'
        return None
    
    def _parse_attributes(self, token: str) -> dict:
        """Parse all attributes from the tag into a dictionary."""
        # Remove < and > brackets
        inner = token[1:-1]
        
        # Remove tag name (manual_label or auto_label)
        if inner.lower().startswith('manual_label'):
            inner = inner[12:].strip()
        elif inner.lower().startswith('auto_label'):
            inner = inner[10:].strip()
        
        # Parse attributes using regex
        # Matches: attr="value" or attr='value'
        attr_pattern = re.compile(r'(\w+)\s*=\s*["\']([^"\']*)["\']')
        attributes = {}
        
        for match in attr_pattern.finditer(inner):
            key, value = match.groups()
            attributes[key] = value
        
        return attributes
    
    def _update_from_token(self, new_token: str):
        """
        Update internal state by re-parsing the new token.
        
        Args:
            new_token: New token string to parse
        """
        self._token = new_token
        self._label_type = self._detect_label_type(new_token)
        self._attributes = self._parse_attributes(new_token)
    
    def is_manual_label(self) -> bool:
        """Return True if this is a manual_label."""
        return self._label_type == 'manual_label'
    
    def is_auto_label(self) -> bool:
        """Return True if this is an auto_label."""
        return self._label_type == 'auto_label'
    
    @property
    def name(self) -> str:
        """Return the labelname attribute value."""
        return self._attributes.get('labelname', '')
    
    @property
    def attributes(self) -> dict:
        """Return dictionary of all attributes (including labelname)."""
        return self._attributes.copy()
    
    @property
    def label_type(self) -> str:
        """Return 'manual_label' or 'auto_label'."""
        return self._label_type
    
    def _rebuild_token(self):
        """Rebuild the HTML token from the current attributes."""
        reconstructed = f"<{self._label_type}"

        for key, value in self._attributes.items():
            reconstructed += f' {key}="{value}"'

        reconstructed += ">"
        self._token = reconstructed

    def set_attribute(self, attribute: str, value: str):
        """
        Update the value of an existing attribute.

        Args:
            attribute: Name of the attribute to modify.
            value: New value.

        Raises:
            KeyError: If the attribute does not exist.
        """
        if attribute not in self._attributes:
            raise KeyError(f"Attribute '{attribute}' does not exist.")

        self._attributes[attribute] = value
        self._rebuild_token()
    
    def to_string(self, remove_attributes: list = None, keep_attributes: list = None):
        """
        Update the token with filtered attributes and update internal state.
        
        Args:
            remove_attributes: List of attribute names to remove. If None, no removal filtering.
            keep_attributes: List of attribute names to keep (all others removed). If None, no keep filtering.
        
        Note: Only one of remove_attributes or keep_attributes should be provided, not both.
              If both are provided, keep_attributes takes precedence.
              This method modifies the object's internal state.
        
        Examples:
            >>> label = HTMLLabel('<manual_label labelname="mention" docid="123" style="color:red" parent="div">')
            
            # Remove specific attributes (modifies the object)
            >>> label.to_string(remove_attributes=['style', 'parent'])
            >>> print(label)  # '<manual_label labelname="mention" docid="123">'
            
            # Keep only specific attributes (modifies the object)
            >>> label.to_string(keep_attributes=['labelname'])
            >>> print(label)  # '<manual_label labelname="mention">'
        """
        # If both provided, keep_attributes takes precedence
        if keep_attributes is not None:
            # Keep only specified attributes
            filtered_attrs = {k: v for k, v in self._attributes.items() if k in keep_attributes}
        elif remove_attributes is not None:
            # Remove specified attributes
            filtered_attrs = {k: v for k, v in self._attributes.items() if k not in remove_attributes}
        else:
            # No filtering, do nothing
            return
        
        self._attributes = filtered_attrs
        self._rebuild_token()
    
    def switch_type(self):
        """
        Switch between manual_label and auto_label types and update internal state.
        
        If the label is manual_label, it becomes auto_label, and vice versa.
        All attributes are preserved. This method modifies the object's internal state.
        
        Examples:
            >>> label = HTMLLabel('<manual_label labelname="mention" docid="123">')
            >>> label.switch_type()
            >>> print(label)  # '<auto_label labelname="mention" docid="123">'
            >>> label.is_auto_label()  # True
            
            >>> label.switch_type()  # Switch back
            >>> print(label)  # '<manual_label labelname="mention" docid="123">'
            >>> label.is_manual_label()  # True
        """
        # Determine the new label type
        new_type = 'auto_label' if self._label_type == 'manual_label' else 'manual_label'
        
        # Reconstruct token with new type
        reconstructed = f'<{new_type}'
        
        # Add all attributes
        for key, value in self._attributes.items():
            reconstructed += f' {key}="{value}"'
        
        reconstructed += '>'
        
        # Update internal state
        self._update_from_token(reconstructed)
    
    def to_simplified(self) -> str:
        """
        Return simplified token format using labelname as tag name, preserving other attributes.
        
        Converts:
            <manual_label labelname="mention" docid="123"> → <mention docid="123">
            <auto_label labelname="title" titletype="main"> → <title titletype="main">
        
        This method removes the manual_label/auto_label wrapper and the labelname attribute,
        but keeps all other attributes. It does NOT modify the object's internal state.
        
        Returns:
            Simplified token string with format <labelname attr="value" ...>
        
        Examples:
            >>> label = HTMLLabel('<manual_label labelname="mention" docid="123" style="color:red">')
            >>> label.to_simplified()
            '<mention docid="123" style="color:red">'
            
            >>> label2 = HTMLLabel('<auto_label labelname="title" titletype="main">')
            >>> label2.to_simplified()
            '<title titletype="main">'
            
            >>> label3 = HTMLLabel('<manual_label labelname="decision">')
            >>> label3.to_simplified()
            '<decision>'
            
            >>> print(label)  # Original token unchanged
            '<manual_label labelname="mention" docid="123" style="color:red">'
        """
        labelname = self._attributes.get('labelname', '')
        
        # Start with the labelname as tag
        simplified = f'<{labelname}'
        
        # Add all attributes except 'labelname'
        for key, value in self._attributes.items():
            if key != 'labelname':
                simplified += f' {key}="{value}"'
        
        simplified += '>'
        return simplified
    
    def __repr__(self):
        return f"HTMLLabel(type={self._label_type}, name={self.name}, attrs={self.attributes})"
    
    def __str__(self):
        return self._token
    


def from_simplified(simplified_token: str, label_type: str = 'auto_label') -> HTMLLabel:
    """
    Convert a simplified token format to a full HTMLLabel object.
    
    Takes a simplified token like <title titletype="main"> and converts it to
    a full label format like <auto_label labelname="title" titletype="main">.
    
    Args:
        simplified_token: Simplified token string like '<title titletype="main">'
        label_type: Either 'manual_label' or 'auto_label' (default: 'auto_label')
    
    Returns:
        HTMLLabel object with the full token format
    
    Raises:
        ValueError: If simplified_token is not a valid tag or label_type is invalid
    
    Examples:
        >>> label = from_simplified('<title titletype="main">', 'manual_label')
        >>> print(label)
        '<manual_label labelname="title" titletype="main">'
        
        >>> label2 = from_simplified('<mention docid="123">', 'auto_label')
        >>> print(label2)
        '<auto_label labelname="mention" docid="123">'
        
        >>> label3 = from_simplified('<decision>')
        >>> print(label3)
        '<auto_label labelname="decision">'
    """
    # Validate label_type
    if label_type not in ['manual_label', 'auto_label']:
        raise ValueError(f"label_type must be 'manual_label' or 'auto_label', got: {label_type}")
    
    # Validate simplified_token format
    if not simplified_token.startswith('<') or not simplified_token.endswith('>'):
        raise ValueError(f"Invalid token format: {simplified_token}")
    
    # Remove < and > brackets
    inner = simplified_token[1:-1].strip()
    
    # Parse tag name and attributes
    # Split on first space to separate tag name from attributes
    tokens = inner.split()

    name_tokens = []
    attr_tokens = []

    for tok in tokens:
        if '=' in tok:
            attr_tokens.append(tok)
        else:
            if not attr_tokens:
                name_tokens.append(tok)
            else:
                # Edge case: malformed token after attributes
                attr_tokens.append(tok)

    tag_name = ' '.join(name_tokens)
    other_attrs = ' '.join(attr_tokens)

    
    # Construct full token
    full_token = f'<{label_type} labelname="{tag_name}"'
    
    if other_attrs:
        full_token += f' {other_attrs}'
    
    full_token += '>'
    
    # Return HTMLLabel object
    return HTMLLabel(full_token)




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
    
    return normalized_tokens