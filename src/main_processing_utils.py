import os
import json
from tqdm import tqdm
from typing import List, Tuple, Optional

from .html_cleaner import clean_tokens
from .tokenizer_utils import decode
from .chunck_level_post_processing import apply_post_processing_transforms
from .prompt_utils import get_prompt_processing, get_prompt_sublabel_extraction
from .levenshtein_utils import distance_lists_auto_label, apply_operations_safe
from .few_shot_utils import prepare_label_tokens, get_list_of_mention
from .verification_utils import verify_processed_chunk


class ProcessingHistory:
    """Track processing history for debugging and analysis."""
    
    def __init__(self):
        self.entries = []
    
    def add(self, status: str, chunk_idx: int, raw_output: str, error_details: str = None):
        """Add an entry to the history."""
        self.entries.append({
            "status": status,
            "chunk_idx": chunk_idx,
            "raw_output": raw_output,
            "error_details": error_details
        })
    
    def save(self, output_dir: str, filename: str):
        """Save history to JSON file."""
        if not output_dir or not filename:
            return
        
        json_path = os.path.join(output_dir, f"history_{filename}.json")
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=4)
            print(f"   ✓ Processing history saved to: {json_path}")
        except Exception as e:
            print(f"   ✗ Error saving history: {e}")
    
    def summary(self) -> dict:
        """Get summary statistics."""
        statuses = [entry["status"] for entry in self.entries]
        return {
            "total": len(statuses),
            "success": statuses.count("Success"),
            "hallucination_fail": statuses.count("Hallucination Fail"),
            "consistency_fail": statuses.count("Consistency Fail"),
            "label_scheme_fail": statuses.count("Label Scheme Fail"),
            "double_hallucination_fail": statuses.count("Double Hallucination Fail"),
            "double_consistency_fail": statuses.count("Double Consistency Fail")
        }




# ==============================================================================================================================
# ============ BELOW ARE MAIN INTERNAL HELPER FUNCTIONS FOR CHUNCK PROCESSING (STAGE 1 OF THE EXTRACTION PROCESS)  =============
# ==============================================================================================================================

def direct_post_processing(chunk, cleaned_chunk, llm_output, label_config, allowed_labels, cot):
    # ------ 3. POST-PROCESS OUTPUT ------
        #try:
        processed_tokens = apply_post_processing_transforms(
            raw_output=llm_output,
            use_simplified=label_config.get("use_simplified", False),
            label_type='auto_label',
            cot=cot
        )
        #except Exception as e:
        #    return chunk, "Post-processing Error", f"Failed to post-process: {str(e)}"
        
        #print(f"   → Processed tokens before verification: {decode(processed_tokens)}")
        
        # ------ 4. APPLY ERROR CORRECTION (BEFORE VERIFICATION) ------
        # This aligns tokens to handle minor discrepancies
        _, operations = distance_lists_auto_label(cleaned_chunk, processed_tokens)
        processed_tokens_corrected = apply_operations_safe(processed_tokens, operations)
        #print(operations) # for debugging
        ##### IGNORE IT FOR TEST !!!!!!!!! ###########
        #processed_tokens_corrected = processed_tokens
        # ------ 5. VERIFY OUTPUT ------

        #print(f"Processed tokens after correction: {decode(processed_tokens_corrected)}")
        verification = verify_processed_chunk(
            original_tokens=cleaned_chunk,
            processed_tokens=processed_tokens_corrected,
            allowed_labels=allowed_labels,
            check_scheme=True
            )
        
        return processed_tokens_corrected, verification
        


def process_single_chunk(
    model,
    chunk: list,
    system_prompt: str,
    user_prompt_template: str,
    label_config: dict,
    allowed_labels: Optional[List[str]] = None,
    cot: bool = False,
    fall_back: bool = False
) -> Tuple[list, str, str]:
    """
    Process a single chunk with post-processing, verification, and fallback.
    
    Pipeline:
    1. Prepare input (clean chunk)
    2. Generate LLM output
    3. Post-process output (extract, transform)
    4. Verify output (hallucination, consistency, label scheme)
    
    Args:
        model: LLM model instance
        chunk: List of tokens to process
        system_prompt: System prompt for LLM
        user_prompt_template: User prompt template with {text} placeholder
        label_config: Configuration for label transformations
        allowed_labels: List of allowed label names for scheme validation
        cot: Whether to use chain-of-thought prompting
        fall_back: Whether to implement a fall-back mechanism in case of verification failure
        max_fallback_attempts: Maximum number of fallback attempts per error type
    
    Returns:
        Tuple of (processed_tokens, status, error_details)
        - processed_tokens: Successfully processed tokens or original chunk if failed
        - status: "Success", "Hallucination Fail", "Consistency Fail", etc.
        - error_details: Description of error if failed, None if success
    """    
    # ------ 1. PREPARE INPUT ------
    cleaned_chunk = prepare_label_tokens(
        chunk,
        label_config={
            "keep_attributes": ["labelname"],
            "switch_type": False,
            "use_simplified": False
        }
    )
    text = decode(cleaned_chunk)
    
    # ------ 2. GENERATE LLM OUTPUT ------
    user_prompt = user_prompt_template.format(text=text)
    raw_output = model.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    print(f"   → Raw LLM output for chunk: {raw_output}...")  
    processed_tokens_corrected, verification = direct_post_processing(chunk=chunk, cleaned_chunk=cleaned_chunk, llm_output=raw_output, label_config=label_config, allowed_labels=allowed_labels, cot=cot)

    if verification.passed:
            return processed_tokens_corrected, "Success", None
    
    returned_chunk = chunk # by default, we return the original chunk if verification fails. But in case of nesting failure, we can try to correct it and return the corrected version even if the verification fails, as it is a minor structural issue that does not affect the content of the labels.
    if verification.error_type == "nesting": # meaning no other error was detected but the nesting check failed. We can try to correct it, but as a minor structural issue, we will return it in case of a second failure
        returned_chunk = processed_tokens_corrected

    # FALL BACK IF ERROR OCCURS
    if fall_back:
        with open(".\\llm_based_annotation\\utils_extraction\\prompts\\fall_back_prompt.txt", 
                "r", 
                encoding="utf-8") as f:
            
            fall_back_system_prompt = f.read()

        llm_output_corrected = decode(prepare_label_tokens(chunk=processed_tokens_corrected, 
                                                        label_config=label_config))
        fall_back_user_prompt = f"""
                    ## Input Example

                    [ORIGINAL PARAGRAPH]
                    {text}
                    ...

                    [ANNOTATED PARAGRAPH]
                    {llm_output_corrected}
                    ...

                    [VERIFICATION]
                    error_type: {verification.error_type}
                    details: {verification.details}

                    ---

                    ## Output

                    (Return only the corrected annotated paragraph)
        """

        verified_output = model.generate(
            system_prompt=fall_back_system_prompt,
            user_prompt=fall_back_user_prompt
        )

        processed_tokens_corrected, verification = direct_post_processing(chunk=chunk, cleaned_chunk=cleaned_chunk, llm_output=verified_output, label_config=label_config, allowed_labels=allowed_labels, cot=cot) 
        
        if verification.passed:
            return processed_tokens_corrected, "Success", None
        
        if verification.error_type == "nesting":
            returned_chunk = processed_tokens_corrected
        

        
        # ------ 6. HANDLE VERIFICATION FAILURES ------
        print(f"   ⚠ Verification failed again: {verification.error_type}")
        print(f"   Details: {verification.details}")



    
    
    return returned_chunk, verification.error_type, verification.details




# ============================================================
# =========== MAIN FUNCTION FOR CHUNCK PROCESSING  ===========
# ============================================================


def process_chunks(
    model,
    token_chunks: list,
    process_prompt_path: str,
    label_config: dict,
    few_shot_examples: Optional[list] = None,
    allowed_labels: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    filename: Optional[str] = None,
    cot: bool = False,
    dynamic_few_shot_selection: bool = False,
    fall_back = False
) -> list:
    """
    Process multiple chunks using AI model with verification and fallback.
    
    This is the main entry point for chunk processing. It coordinates:
    - LLM generation
    - Post-processing transformations
    - Verification checks
    - Error correction and fallbacks
    - History tracking
    
    Args:
        model: LLM model instance
        token_chunks: List of token chunks to process
        process_prompt_path: Path to the main processing prompt file
        label_config: Configuration for label transformations
        few_shot_examples: Optional list of (input, output) examples
        allowed_labels: Optional list of allowed label names
        output_dir: Optional directory to save outputs
        filename: Optional filename prefix for outputs
        max_fallback_attempts: Maximum fallback attempts per error type
    
    Returns:
        List of processed token chunks
    """
    # Load prompts

    system_prompt, user_prompt_template = get_prompt_processing(
        prompt_path=process_prompt_path, 
        few_shot_examples=few_shot_examples, 
        chunks=token_chunks, 
        dynamic_few_shot_selection=dynamic_few_shot_selection
    )


    # Initialize tracking
    processed_chunks = []
    history = ProcessingHistory()
    
    print(f"   ✓ Processing {len(token_chunks)} chunks with LLM...")
    print(f"   ✓ Using {len(few_shot_examples) if few_shot_examples else 0} few-shot examples")
    if allowed_labels:
        print(f"   ✓ Label scheme validation enabled with {len(allowed_labels)} allowed labels")
    
    # Process each chunk
    for idx, chunk in enumerate(tqdm(token_chunks, desc="Processing chunks")):

        processed_tokens, status, error_details = process_single_chunk(
            model=model,
            chunk=chunk,
            system_prompt=system_prompt[idx] if dynamic_few_shot_selection else system_prompt,
            user_prompt_template=user_prompt_template[idx] if dynamic_few_shot_selection else user_prompt_template,
            label_config=label_config,
            allowed_labels=allowed_labels,
            cot=cot,
            fall_back = fall_back
        )
    
        processed_chunks.append(processed_tokens)
        history.add(status, idx, decode(processed_tokens), error_details)
        
        if status != "Success" and not status.startswith("Success (after"):
            print(f"   ⚠ Chunk {idx} failed: {status}")
    
    # Save history and results
    history.save(output_dir, filename)
    
    # Print summary
    summary = history.summary()
    print(f"\n   ✓ Processing completed:")
    print(f"      - Total chunks: {summary['total']}")
    print(f"      - Successful: {summary['success']}")
    print(f"      - Failed: {summary['total'] - summary['success']}")
    
    if output_dir and filename:
        json_path = os.path.join(output_dir, f"processed_chunks_{filename}.json")
        try:
            # Convert token lists to strings for JSON serialization
            chunks_as_strings = [decode(chunk) for chunk in processed_chunks]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(chunks_as_strings, f, indent=4, ensure_ascii=False)
            print(f"   ✓ Processed chunks saved to: {json_path}")
        except Exception as e:
            print(f"   ✗ Error saving processed chunks: {e}")
    
    return processed_chunks








# ==============================================================================================================================
# =========== BELOW ARE MAIN INTERNAL HELPER FUNCTIONS FOR SUBLABELS PROCESSING (STAGE 2 OF THE EXTRACTION PROCESS)  ===========
# ==============================================================================================================================


def _build_processing_segments(tokens, parent_mentions):
    """
    Build a list of segments alternating between:
      - non-processable token spans
      - processable mention spans

    Returns:
        List[dict]: each dict has:
            - "process": bool
            - "tokens": list
            - "meta": optional mention metadata
    """
    segments = []
    cursor = 0

    for html_label, start_idx, end_idx in parent_mentions:
        # Non-processable tokens before the mention
        if cursor < start_idx:
            segments.append({
                "process": False,
                "tokens": tokens[cursor:start_idx]
            })

        # The mention itself (processable)
        segments.append({
            "process": True,
            "tokens": tokens[start_idx:end_idx + 1],
            "meta": {
                "label": html_label,
                "start": start_idx,
                "end": end_idx
            }
        })

        cursor = end_idx + 1

    # Trailing non-processable tokens
    if cursor < len(tokens):
        segments.append({
            "process": False,
            "tokens": tokens[cursor:]
        })

    return segments


def process_single_mention(
    model,
    mention: list,
    system_prompt: str,
    user_prompt_template: str,
    sublabel_config: dict,
    allowed_labels: list = None,
    context: str = None,
):
    """
    Process a single parent mention to extract sublabels.
    
    Pipeline:
    1. Prepare input (simplified form, keep right attributes)
    2. Decode mention to text
    3. Generate LLM output
    4. Post-process output (TODO: extract, transform)
    5. Verify output (hallucination, consistency, label scheme)
    6. If verification fails, apply fallback
    7. If still fails, return original mention
    
    Args:
        model: LLM model instance
        mention: List of tokens for the parent mention
        system_prompt: System prompt for LLM
        user_prompt_template: User prompt template with {text} placeholder
        sublabel_config: Configuration for sublabel transformations
        allowed_labels: List of allowed sublabel names
        max_fallback_attempts: Maximum number of fallback attempts
        context: Contextual information for the LLM
    Returns:
        Tuple of (processed_tokens, status, error_details)
    """
    # ------ 1. PREPARE INPUT ------
    input_config = sublabel_config.copy()
    input_config["keep_labels"] = None # Keep everything
    input_config["switch_type"] = False # Keep original types for input
    prepared_mention = prepare_label_tokens(mention,
        label_config={
            "switch_type": False,
            "use_simplified": sublabel_config.get("use_simplified", False),
            "keep_attributes": sublabel_config.get("keep_attributes", None)
        }
    )

    
    # ------ 2. DECODE TO TEXT ------
    text = decode(prepared_mention)
    
    # ------ 3. GENERATE LLM OUTPUT ------
    if context:
        user_prompt = user_prompt_template.format(text=text, context=context)
    else:
        user_prompt = user_prompt_template.format(text=text)
    raw_output = model.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    
    # ------ 4. POST-PROCESS OUTPUT ------
    try:
        processed_tokens = apply_post_processing_transforms(
            raw_output=raw_output,
            use_simplified=sublabel_config.get("use_simplified", False), # Did we use simplified form ?
            label_type='auto_label'
        )
    except Exception as e:
        return mention, "Post-processing Error", f"Failed to post-process: {str(e)}"
    
    
    # ------ 5. APPLY ERROR CORRECTION ------
    # This aligns tokens to handle minor discrepancies
    cleaned_input_mention_token = prepare_label_tokens(
        mention,
        label_config={
            "keep_attributes": sublabel_config.get("keep_attributes", None),
            "switch_type": False,
            "use_simplified": False
        }
    ) # Just remove non-kept attributes for alignment
    _, operations = distance_lists_auto_label(cleaned_input_mention_token, processed_tokens)
    processed_tokens_corrected = apply_operations_safe(processed_tokens, operations)
    


    
    # ------ 6. VERIFY OUTPUT ------
    verification = verify_processed_chunk(
        original_tokens=mention,
        processed_tokens=processed_tokens_corrected,
        allowed_labels=allowed_labels,
        check_scheme=True
    )
    
    if verification.passed:
        return processed_tokens_corrected, "Success", None
    
    else :
        return mention, "Verification Failed", verification.details



# ============================================================
# ========== MAIN FUNCTION FOR SUBLABEL PROCESSING  ==========
# ============================================================
def process_labels(
    model,
    tokens: list,
    sublabel_config: dict,
    few_shot_examples: list = None,
    prompt_path: str = None,
    sublabel_definitions=None,
    output_dir: str = None,
    filename: str = None,
    with_context: int = None,
):
    """
    Process tokens to extract sublabels from parent mentions.
    
    This function extracts sublabels (e.g., title) from already annotated
    parent labels (e.g., decision, legislation, secondary sources).
    
    Args:
        model: LLM model instance
        tokens: List of tokens containing parent labels
        sublabel_config: Configuration dict with:
            - parent: List of parent label names
            - keep_labels: List of sublabel names to extract
            - keep_attributes, switch_type, use_simplified: Transform options
        few_shot_examples: Optional list of (input, output) examples
        prompt_path: Optional path to prompt templates
        output_dir: Optional directory to save outputs
        filename: Optional filename prefix for outputs
        max_fallback_attempts: Maximum fallback attempts per error type
    
    Returns:
        List of processed tokens (flat list, not chunked)
    """
    # ------ 1. GET PROMPT ------
    system_prompt, user_prompt_template = get_prompt_sublabel_extraction(
        prompt_path=prompt_path,
        keep_labels=sublabel_config["new_labels"],
        few_shot_examples=few_shot_examples,
        sublabel_definitions=sublabel_definitions,
        with_context_bool= True if with_context else False
    )
    
    # ------ 2. GET LIST OF AUTO_LABEL MENTIONS TO PROCESS ------
    # We're looking for auto_label parents (already extracted from previous step)
    parent_mentions = get_list_of_mention(
        tokens=tokens,
        keep_labels=sublabel_config["parent"],
        label_type="auto_label"  # Process auto_labels from parent extraction
    )
    
    print(f"   ✓ Found {len(parent_mentions)} parent mentions to process")
    
    # ------ 3. INITIALIZE TRACKING ------
    history = ProcessingHistory()
    
    # Create a copy of tokens to modify
    processed_tokens = tokens.copy()
    
    # ------ 4. BUILD SEGMENTS ------
    segments = _build_processing_segments(tokens, parent_mentions)

    print(f"   ✓ Built {len(segments)} token segments "
        f"({sum(s['process'] for s in segments)} to process)")

    # ------ 5. PROCESS EACH PROCESSABLE SEGMENT ------
    for idx, segment in enumerate(tqdm(segments, desc="Processing mentions")):
        if not segment["process"]:
            continue

        mention = segment["tokens"]
        html_label = segment["meta"]["label"]



        context = None
        if with_context:
            context = decode(clean_tokens(tokens[max(0, segment["meta"]["start"] - with_context): segment["meta"]["start"]]))

        #print(context) # for debugging

        processed_mention, status, error_details = process_single_mention(
            model=model,
            mention=mention,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            context=context,
            sublabel_config=sublabel_config,
            allowed_labels=sublabel_config["new_labels"] + sublabel_config["already_labeled"] + sublabel_config["parent"],
        )

        # Replace the entire segment safely
        segment["tokens"] = processed_mention

        # Track history
        history.add(
            status,
            idx,
            decode(processed_mention),
            error_details
        )

        if status != "Success" and not status.startswith("Success (after"):
            print(f"   ⚠ Segment {idx} ({html_label.name}) failed: {status} \n Details: {error_details}")

    
    # ------ 6. SAVE HISTORY AND RESULTS ------
    if output_dir and filename:
        history.save(output_dir, f"{filename}_sublabel")
        
        # Save processed tokens
        json_path = os.path.join(output_dir, f"processed_sublabels_{filename}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(decode(processed_tokens), f, indent=4, ensure_ascii=False)
            print(f"   ✓ Processed tokens saved to: {json_path}")
        except Exception as e:
            print(f"   ✗ Error saving processed tokens: {e}")
    
    # ------ 7. PRINT SUMMARY ------
    summary = history.summary()
    print(f"\n   ✓ Sublabel extraction completed:")
    print(f"      - Total mentions: {summary['total']}")
    print(f"      - Successful: {summary['success']}")
    print(f"      - Failed: {summary['total'] - summary['success']}")



    # ------ 8. FLATTEN SEGMENTS ------
    processed_tokens = [
        token
        for segment in segments
        for token in segment["tokens"]
    ]
    
    return processed_tokens