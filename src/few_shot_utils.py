from .html_cleaner import clean_tokens
from .tokenizer_utils import decode
from .htmlLabel import HTMLLabel, from_simplified
from .html_utils import is_manual_label_tag, is_auto_label_tag
from .tokenizer_utils import tokenize
import random

def prepare_label_tokens(chunk, label_config):
    """
    Transform label tokens according to specified parameters.
    
    Handles both opening and closing tags, using a stack to track label names
    for proper closing tag generation in simplified mode.
    
    Args:
        chunk: List of tokens to process
        remove_attributes: List of attribute names to remove from labels
        keep_attributes: List of attribute names to keep in labels
        switch_type: If True, switch between manual_label and auto_label types
        use_simplified: If True, output simplified form (e.g., <title> instead of <manual_label labelname="title">)
        keep_labels: List of label names to keep (all others removed). If None, keep all labels.
        remove_labels: List of label names to remove. If None, no removal filtering.
    
    Returns:
        List of transformed tokens
    """

    # Get config parameters
    remove_attributes = label_config.get('remove_attributes', None)
    keep_attributes = label_config.get('keep_attributes', None)
    switch_type = label_config.get('switch_type', False)
    use_simplified = label_config.get('use_simplified', False)
    keep_labels = label_config.get('keep_labels', None)
    remove_labels = label_config.get('remove_labels', None)


    transformed_chunk = []
    label_name_stack = []  # Stack to track opened label names for closing tags
    skip_stack = []  # Stack to track which labels are being skipped (removed)
    
    for token in chunk:
        # Check if token is a label opening tag
        is_manual_open = token.lower().startswith('<manual_label') and token.endswith('>')
        is_auto_open = token.lower().startswith('<auto_label') and token.endswith('>')
        
        # Check if token is a label closing tag
        is_manual_close = token.lower().startswith('</manual_label') and token.endswith('>')
        is_auto_close = token.lower().startswith('</auto_label') and token.endswith('>')
        
        if is_manual_open or is_auto_open:
            # Process opening tag
            try:
                label = HTMLLabel(token)
                
                # Check if this label should be kept based on keep_labels/remove_labels
                should_keep = True
                if keep_labels is not None:
                    should_keep = label.name in keep_labels
                elif remove_labels is not None:
                    should_keep = label.name not in remove_labels
                
                if not should_keep:
                    # Skip this label tag, but track it for closing tag
                    skip_stack.append(True)
                    # Don't add to transformed_chunk (removes the tag but keeps content)
                    continue
                
                skip_stack.append(False)
                
                # Apply attribute filtering if specified
                if keep_attributes is not None or remove_attributes is not None:
                    label.to_string(remove_attributes=remove_attributes, keep_attributes=keep_attributes)
                
                # Switch type if requested
                if switch_type:
                    label.switch_type()
                
                # Output simplified or full form
                if use_simplified:
                    transformed_token = label.to_simplified()
                    # Push label name to stack for closing tag
                    label_name_stack.append(label.name)
                else:
                    transformed_token = str(label)
                
                transformed_chunk.append(transformed_token)
            except ValueError:
                # If token can't be parsed, keep as-is
                transformed_chunk.append(token)
                
        elif is_manual_close or is_auto_close:
            # Process closing tag
            # Check if we skipped the corresponding opening tag
            if skip_stack:
                was_skipped = skip_stack.pop()
                if was_skipped:
                    # Skip this closing tag too (removes the tag but keeps content)
                    continue
            
            if use_simplified:
                # Pop label name from stack
                if label_name_stack:
                    label_name = label_name_stack.pop()
                    transformed_token = f'</{label_name}>'
                else:
                    # Stack empty, keep as-is (shouldn't happen with valid HTML)
                    transformed_token = token
            else:
                # Not simplified: handle type switching if needed
                if switch_type:
                    # Switch closing tag type
                    if is_manual_close:
                        transformed_token = '</auto_label>'
                    else:
                        transformed_token = '</manual_label>'
                else:
                    transformed_token = token
            
            transformed_chunk.append(transformed_token)
        else:
            # Not a label tag, keep as-is
            transformed_chunk.append(token)
    
    return transformed_chunk


def extract_few_shot_examples(token_chunks, label_config):
    """
    Extract few-shot examples from the chunks with flexible label transformation.
    
    Args:
        token_chunks: List of token chunks to process
        remove_attributes: List of attribute names to remove from labels. If None, no removal filtering.
        keep_attributes: List of attribute names to keep in labels (all others removed). If None, no keep filtering.
        switch_type: If True, switch between manual_label and auto_label types
        use_simplified: If True, output simplified form (e.g., <title> instead of <manual_label labelname="title">)
        keep_labels: List of label names to keep (all others removed). If None, keep all labels.
        remove_labels: List of label names to remove. If None, no removal filtering.
    
    Returns:
        list: List of tuples (input_tokens, output_tokens)
              - input_tokens: cleaned tokens without any label tags
              - output_tokens: tokens with transformed label tags according to parameters
    
    Examples:
        >>> # Remove specific attributes and switch to auto_label
        >>> examples = extract_few_shot_examples(chunks, remove_attributes=['style', 'parent'], switch_type=True)
        
        >>> # Keep only labelname and docid attributes
        >>> examples = extract_few_shot_examples(chunks, keep_attributes=['labelname', 'docid'])
        
        >>> # Output simplified form with proper closing tags
        >>> examples = extract_few_shot_examples(chunks, use_simplified=True)
        >>> # <manual_label labelname="title">Text</manual_label> → <title>Text</title>
        
        >>> # Keep only decision labels (removes title tags but keeps text)
        >>> examples = extract_few_shot_examples(chunks, keep_labels=['decision'], use_simplified=True)
        >>> # <decision><title>John Campbell Law Corporation v.</title></decision>
        >>> # → <decision>John Campbell Law Corporation v.</decision>
        
        >>> # Remove title labels (keeps all other labels)
        >>> examples = extract_few_shot_examples(chunks, remove_labels=['title'], use_simplified=True)
        >>> # Same result as above
    """
    examples = []
    
    for chunk in token_chunks:
        # Input: tokens cleaned from all label tags
        input_chunk = clean_tokens(chunk, normalize=True, keep_manual_label=False, 
                                   keep_auto_label=False, keep_bookmarks=False)
        
        # Output: tokens with transformed label tags
        output_chunk = prepare_label_tokens(
            chunk, 
            label_config
        )
        
        examples.append((decode(input_chunk), decode(output_chunk)))
    
    print(f"   ✓ Extracted {len(examples)} few-shot examples from chunks")
    return examples


def select_few_shot(examples, n, method="order", list_of_labels=None, distribution=None):
    """
    Select n few-shot examples from the provided list.
    
    Args:
        examples: List of tuples (input, expected_output)
        n: Number of examples to select
        method: Selection method - "order", "random", or "distributed"
        list_of_labels: List of label names for distributed selection (e.g., ["source"])
        distribution: List of proportions for each label (e.g., [0.5])
                     If sum < 1.0, remainder is filled with random "other" examples
    
    Returns:
        list: Selected few-shot examples
        
    Examples:
        # Select first 10 in order
        select_few_shot(examples, 10, method="order")
        
        # Select 10 with 50% containing "source" label, 50% random others
        select_few_shot(examples, 10, method="distributed", 
                       list_of_labels=["source"], distribution=[0.5])
        
        # Select 10 with 40% source, 30% title, 30% random others
        select_few_shot(examples, 10, method="distributed",
                       list_of_labels=["source", "title"], distribution=[0.4, 0.3])
    """
    if method == "order":
        if n >= len(examples):
            return examples
        else:
            return examples[:n]
    
    if method == "random":
        if n >= len(examples):
            return examples
        else:
            return random.sample(examples, n)
    
    if method == "distributed":
        if not list_of_labels or not distribution:
            raise ValueError("method='distributed' requires list_of_labels and distribution parameters")
        
        if len(distribution) != len(list_of_labels):
            raise ValueError(f"distribution length ({len(distribution)}) must match list_of_labels length ({len(list_of_labels)})")
        
        dist_sum = sum(distribution)
        if dist_sum > 1.0:
            raise ValueError(f"distribution sum cannot exceed 1.0, got {dist_sum}")
        
        # Categorize examples by labels
        categorized = {label: [] for label in list_of_labels}
        categorized["other"] = []
        
        for example in examples:
            _, output_text = example
            output_tokens = tokenize(output_text)
            labels_in_example = []
            for token in output_tokens:
                if is_auto_label_tag(token) == 1:
                    token_label = HTMLLabel(token)
                    labels_in_example.append(token_label.name)
                
                elif token.startswith('<') and token.endswith('>'):
                    # Could be a simplified tag
                    simple_label = from_simplified(token)
                    labels_in_example.append(simple_label.name)
            
            
            # Check if example contains any of the target labels
            found = False
            for target_label in list_of_labels:
                if target_label in labels_in_example:
                    categorized[target_label].append(example)
                    found = True
                    break  # Only categorize by first matching label
            
            if not found:
                categorized["other"].append(example)
        
        # Select examples according to distribution
        selected = []
        
        # First, select from specified labels
        for i, label in enumerate(list_of_labels):
            count = int(n * distribution[i])
            available = categorized[label]
            
            if count > len(available):
                print(f"   ⚠ Warning: Requested {count} examples with label '{label}', but only {len(available)} available")
                selected.extend(available)
            else:
                selected.extend(random.sample(available, count))
        
        # Fill remaining with "other" (automatically if distribution sum < 1.0)
        remaining = n - len(selected)
        if remaining > 0:
            available_other = categorized["other"]
            if remaining > len(available_other):
                selected.extend(available_other)
            else:
                selected.extend(random.sample(available_other, remaining))
        
        # Shuffle to mix the categories
        random.shuffle(selected)
        
        return selected[:n]  # Ensure we return exactly n examples
    
    raise ValueError(f"Unknown method: {method}")
    



def get_list_of_mention(tokens, keep_labels, label_type=None):
    """
    Extract mentions from tokens and return their positions.
    
    Args:
        tokens: List of tokens to search
        keep_labels: List of label names to keep (e.g., ["title", "decision"])
        label_type: Optional filter - "manual_label", "auto_label", or None (both)
    
    Returns:
        List of tuples: (HTMLLabel object, start_index, end_index)
        - HTMLLabel object: The parsed opening tag
        - start_index: Index of the opening tag in tokens list
        - end_index: Index of the closing tag in tokens list
    """
    mentions = []
    i = 0
    
    while i < len(tokens):
        token = tokens[i]
        
        # Check if this matches the label type we're looking for
        is_match = False
        if label_type == "manual_label" and is_manual_label_tag(token) == 1:
            is_match = True
        elif label_type == "auto_label" and is_auto_label_tag(token) == 1:
            is_match = True
        elif label_type is None and (is_manual_label_tag(token) == 1 or is_auto_label_tag(token) == 1):
            is_match = True
        
        if is_match:
            html_label = HTMLLabel(token)
            
            # Check if this label is in keep_labels
            if html_label.name in keep_labels:
                start_index = i
                depth = 1
                i += 1
                
                # Find the matching closing tag
                while i < len(tokens) and depth > 0:
                    current_token = tokens[i]
                    
                    # Check if it's an opening tag of the same type
                    if label_type == "manual_label" and is_manual_label_tag(current_token) == 1:
                        depth += 1
                    elif label_type == "auto_label" and is_auto_label_tag(current_token) == 1:
                        depth += 1
                    elif label_type is None:
                        if is_manual_label_tag(current_token) == 1 or is_auto_label_tag(current_token) == 1:
                            depth += 1
                    
                    # Check if it's a closing tag of the same type
                    if label_type == "manual_label" and is_manual_label_tag(current_token) == 2:
                        depth -= 1
                    elif label_type == "auto_label" and is_auto_label_tag(current_token) == 2:
                        depth -= 1
                    elif label_type is None:
                        if is_manual_label_tag(current_token) == 2 or is_auto_label_tag(current_token) == 2:
                            depth -= 1
                    
                    if depth == 0:
                        end_index = i
                        mentions.append((html_label, start_index, end_index))
                        break
                    
                    i += 1
                continue
        
        i += 1
    
    return mentions


def extract_few_shot_examples_from_labels(tokens, sublabel_config):
    """
    Extract few-shot examples from parent labels containing sublabels.
    
    Args:
        tokens: List of tokens to process
        sublabel_config: Configuration dict with:
            - parent: List of parent label names to extract from
            - keep_labels: List of sublabel names to keep in output
            - keep_attributes, switch_type, use_simplified: Transform options
    
    Returns:
        List of tuples (input, output) where:
        - input: Parent label with only parent tag preserved
        - output: Parent label with both parent and specified sublabels preserved
    """
    examples = []
    
    # Get all mentions of parent labels using the utility function
    parent_mentions = get_list_of_mention(tokens=tokens, keep_labels=sublabel_config["parent"], label_type="manual_label")
    
    for _, start_idx, end_idx in parent_mentions:
        # Extract the mention tokens (from start to end inclusive)
        mention = tokens[start_idx:end_idx + 1]
        
        # Input: keep only parent labels
        input_legal_config = sublabel_config.copy()
        input_legal_config["keep_labels"] = sublabel_config["parent"] + sublabel_config["already_labeled"]
        input_tokens = prepare_label_tokens(mention, label_config=input_legal_config)
        
        # Output: keep both parent and sublabels
        output_legal_config = sublabel_config.copy()
        output_legal_config["keep_labels"] = sublabel_config["new_labels"] + sublabel_config["already_labeled"] + sublabel_config["parent"]
        output_tokens = prepare_label_tokens(mention, label_config=output_legal_config)
        
        examples.append((decode(input_tokens), decode(output_tokens)))
    
    return examples