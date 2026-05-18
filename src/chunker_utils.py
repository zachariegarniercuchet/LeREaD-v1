import itertools
import re 
from .tokenizer_utils import decode

def chunk_tokens(tokens: list, min_tokens: int = 500, stop_bookmark_separation=False) -> list:
    """
    Chunk a list of tokens into chunks of at least min_tokens.
    
    STRICT RULES:
    1. Accumulate tokens until reaching min_tokens
    2. After min_tokens, continue ONLY until:
        - Quotations are closed (not inside_quotes)
        - AND <manual_label> tags are closed (depth == 0)
        - AND current token is a word (not whitespace/punctuation)
    
    If stop_bookmark_separation=True, splits at bookmark first, then chunks each part separately.
    
    Returns list of token chunks, or tuple of two lists if bookmark separation.
    """
    
    # If stop_bookmark_separation is True, find and split at bookmark
    if stop_bookmark_separation:
        bookmark_idx = None
        for i in range(len(tokens) - 2):
            if (tokens[i] == '<htmllabelizer_bookmark id="stop">' and 
                tokens[i+1] == '🔖' and 
                tokens[i+2] == '</htmllabelizer_bookmark>'):
                bookmark_idx = i
                break
        
        if bookmark_idx is not None:
            # Split tokens into before and after (excluding the 3 bookmark tokens)
            tokens_before = tokens[:bookmark_idx]
            tokens_after = tokens[bookmark_idx + 3:]
            
            print(f"   ✓ Found bookmark separator at index {bookmark_idx}")
            print(f"   ✓ Splitting: {len(tokens_before)} tokens before, {len(tokens_after)} tokens after")
            
            # Chunk both parts separately
            chunks_before = _chunk_token_list(tokens_before, min_tokens)
            chunks_after = _chunk_token_list(tokens_after, min_tokens)
            
            print(f"   ✓ Total chunks: {len(chunks_before)} before + {len(chunks_after)} after = {len(chunks_before) + len(chunks_after)}")
            return chunks_before, chunks_after
        else:
            print(f"   ⚠ Warning: stop_bookmark_separation=True but bookmark not found")
    
    # Default behavior: chunk entire list
    return _chunk_token_list(tokens, min_tokens)

def _chunk_token_list(tokens: list, min_tokens: int) -> list:
    """
    Internal method to chunk a token list.
    
    Algorithm:
    1. Accumulate tokens until count >= min_tokens
    2. After min_tokens, continue until ALL conditions are met:
        - NOT inside quotes
        - manual_label depth == 0
        - current token is a word
    """
    chunks = []
    current = []
    count = 0
    
    inside_quotes = False
    manual_label_depth = 0
    
    def is_word_token(tok: str) -> bool:
        """Check if token is a word (alphanumeric, >=1 char)."""
        return tok and tok.strip() and re.match(r'^\w+$', tok) is not None
    
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        current.append(tok)
        count += 1
        
        # Track quote state - ONLY opening quote " starts, ONLY closing quote " ends
        if tok == '“':  # U+201C - LEFT DOUBLE QUOTATION MARK (opening)
            inside_quotes = True
        elif tok == '”':  # U+201D - RIGHT DOUBLE QUOTATION MARK (closing)
            inside_quotes = False
        
        # Track manual_label tag depth
        if tok.startswith('<manual_label'):
            manual_label_depth += 1
        elif tok.startswith('</manual_label'):
            manual_label_depth -= 1
        
        # Check if we can end the chunk
        if count >= min_tokens:
            # Check ALL conditions:
            # 1. Not inside quotes
            # 2. manual_label depth is 0
            # 3. Current token is a word
            if not inside_quotes and manual_label_depth == 0 and is_word_token(tok):
                # All conditions met: end chunk here
                chunks.append(current)
                current = []
                count = 0
                # Reset states for safety
                inside_quotes = False
                manual_label_depth = 0
        
        i += 1
    
    # Append remaining tokens as final chunk
    if current:
        chunks.append(current)
    
    print(f"   ✓ Chunked tokens into {len(chunks)} chunks (>= {min_tokens} tokens each)")
    return chunks




def flatten_token_chunks(token_chunks: list[list[str]], separator: str = None) -> list[str]:
    """
    Flatten a list of token chunks (list of lists) back into a single token list.
    Preserves token order exactly.
    
    Args:
        token_chunks: List of token chunks to flatten
        separator: Optional separator token to insert between chunks (e.g., '<sep>')
    """
    if separator:
        flat = []
        for i, chunk in enumerate(token_chunks):
            flat.extend(chunk)
            if i < len(token_chunks) - 1:  # Don't add separator after last chunk
                flat.append(separator)
    else:
        flat = list(itertools.chain.from_iterable(token_chunks))
    
    print(f"   ✓ Flattened {len(token_chunks)} chunks into {len(flat)} tokens")
    return flat







######################## SECOND CHUNKING VERSION BASED ON SENTENCES ########################
def calculate_combined_density(sentence):
    """Calculate combined period and number density for a sentence."""
    if len(sentence) < 10:
        return 0
    
    num_periods = sentence.count('.')
    num_digits = sum(c.isdigit() for c in sentence)
    
    period_density = (num_periods / len(sentence)) * 100
    digit_density = (num_digits / len(sentence)) * 100
    
    return period_density + digit_density

def detect_citation_sections_sequential(sentences, threshold=25, consecutive_gap=3):
    """
    Detect citation sections using sequential analysis.
    
    Algorithm:
    1. Go through sentences one by one
    2. When density > threshold, start a citation section
    3. Continue until we find 'consecutive_gap' sentences below threshold
    4. End the citation section 'consecutive_gap' sentences before
    5. STOP - all remaining sentences are NOT citations
    
    Args:
        sentences: List of sentences
        threshold: Combined density threshold (%) to consider citation
        consecutive_gap: Number of consecutive non-citation sentences to end section
    
    Returns:
        List of booleans indicating if each sentence is in a citation section
    """
    is_citation = [False] * len(sentences)
    in_citation_section = False
    citation_start = None
    below_threshold_count = 0
    citation_section_ended = False  # Track if we've already found and ended a citation section
    
    for i, sent in enumerate(sentences):
        # If citation section has already ended, all remaining sentences are NOT citations
        if citation_section_ended:
            break
        
        density = calculate_combined_density(sent)
        
        if density > threshold:
            # This is a citation sentence
            if not in_citation_section:
                # Start new citation section
                in_citation_section = True
                citation_start = i
            # Reset the gap counter
            below_threshold_count = 0
            is_citation[i] = True
            
        else:
            # Below threshold
            if in_citation_section:
                # We're in a citation section, count consecutive non-citations
                below_threshold_count += 1
                
                if below_threshold_count >= consecutive_gap:
                    # End citation section: go back 'consecutive_gap' sentences
                    citation_end = i - consecutive_gap
                    # Mark all sentences in this section
                    for j in range(citation_start, citation_end + 1):
                        is_citation[j] = True
                    # STOP HERE - citation section has ended
                    citation_section_ended = True
                    in_citation_section = False
    
    # Handle case where document ends while in citation section
    if in_citation_section and citation_start is not None:
        citation_end = len(sentences) - 1 - below_threshold_count
        for j in range(citation_start, citation_end + 1):
            is_citation[j] = True
    
    return is_citation

def merge_sentences_with_heuristics_tokens(tokens, citation_threshold=25, min_tokens=500):
    """
    Merge sentences based on boundary heuristics, working with tokens.
    
    Input : Flat list of tokens with <sep> as sentence boundaries.
    Output: Flat list of tokens with selective <sep> removal based on citation detection.

    Rules:
    - Normal section: Keep <sep> only if sentence ends with '.'
    - Citation section: Keep <sep> only if sentence ends with ';'
    - Otherwise: Merge sentences together (remove <sep>)
    
    Args:
        tokens: Flat List of token containing <sep> as sentence boundaries
        citation_threshold: Combined density threshold (%) for citation detection
        min_tokens: Minimum number of tokens to consider a sentence

    Returns:
        Flat list of tokens with selective <sep> removal based on citation detection
    """
    if not tokens:
        return []
    
    # STEP 1: Split into sentences based on <sep>
    sentences_tokens = []
    current_sentence = []
    for token in tokens:
        if token == '<sep>':
            if current_sentence:
                sentences_tokens.append(current_sentence)
                current_sentence = []
        else:
            current_sentence.append(token)
    
    if current_sentence:
        sentences_tokens.append(current_sentence)

    # STEP 2: Convert to text for citation detection
    sentences_text = [decode(sent) for sent in sentences_tokens]

    # STEP 3: Detect citations section
    is_citation = detect_citation_sections_sequential(sentences_text, threshold=citation_threshold)

    # STEP 4: Reconstruct flat token list with selective <sep> removal
    result = []
    current_chunk_size = 0  # Track tokens since last <sep>
    
    for i, sent_tokens in enumerate(sentences_tokens):
        result.extend(sent_tokens)
        current_chunk_size += len(sent_tokens)

        # Decide wheter to add <sep> after this sentence
        if i < len(sentences_tokens) - 1:  # Not the last sentence
            should_keep_sep = False

            # Get the last token of current sentence
            last_token = sent_tokens[-1] if sent_tokens else ''

            # Only consider keeping <sep> if we have at least min_tokens accumulated
            if current_chunk_size >= min_tokens:
                if is_citation[i]:
                    # Citation section: keep <sep> only if last token is with ';'
                    if last_token == ";":
                        should_keep_sep = True
                else:
                    # Normal section: keep <sep> only if last token is with '.'
                    if last_token == ".":
                        should_keep_sep = True
            
            if should_keep_sep:
                result.append('<sep>')
                current_chunk_size = 0  # Reset chunk size after keeping <sep>
    
    return result