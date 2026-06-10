"""
Output Processor: Handles LLM output post-processing, error correction, and verification.
This module orchestrates the transformation pipeline: tokenization → extraction → format conversion → verification
"""

from typing import Tuple, Optional, List
from src.htmlLabel import from_simplified
from src.html_utils import get_tag_name, is_auto_label_tag, is_closing_tag, is_opening_tag
from .protected_levenshtein_alignnment import protected_levenshtein_distance, apply_operations_safe
from .verification import verify_processed_chunk, VerificationResult
from ..tokenizer_utils import decode
from src.tokenizer_utils import tokenize


class OutputProcessor:
    """
    Processes raw LLM output through post-processing, correction, and verification steps.
    
    Pipeline:
    1. Post-process: tokenize, extract markers, convert format
    2. Correct: apply Levenshtein alignment to fix minor discrepancies
    3. Verify: check hallucination, consistency, label scheme, nesting
    """
    
    def __init__(self):
        """Initialize the processor."""
        pass
    
    def process(
        self,
        raw_llm_output: str,
        original_chunk: list,
        allowed_labels: Optional[List[str]] = None,
    ) -> Tuple[list, VerificationResult]:
        """
        Process raw LLM output through the complete pipeline.
        
        Args:
            raw_llm_output: Raw string output from LLM
            original_chunk: Original token list (for verification reference)
            allowed_labels: Optional list of allowed label names for validation
        
        Returns:
            Tuple of (processed_tokens, verification_result)
            - processed_tokens: Corrected and processed token list
            - verification_result: VerificationResult with pass/fail info
        
        Example:
            >>> processor = OutputProcessor()
            >>> tokens, result = processor.process(
            ...     raw_llm_output="<auto_label...>text</auto_label>",
            ...     original_chunk=chunk,
            ...     allowed_labels=allowed_labels
            ... )
            >>> if result.passed:
            ...     print("Success!")
        """
        # Step 1: Post-process (tokenize)
        processed_tokens = self._post_process(raw_llm_output, allowed_labels)
        
        # Step 2: Apply error correction (Levenshtein alignment)
        corrected_tokens = self._apply_correction(processed_tokens, original_chunk)
        
        # Step 3: Verify
        verification = verify_processed_chunk(
            original_tokens=original_chunk,
            processed_tokens=corrected_tokens,
            allowed_labels=allowed_labels,
        )
        
        return corrected_tokens, verification
    
    def _post_process(
        self,
        raw_output: str,
        allowed_labels: Optional[List[str]] = None
    ) -> list:
        """
        Apply post-processing transformations to raw output.
        
        Pipeline:
        1. Tokenize raw output
        
        Returns:
            List of processed tokens
        """
        processed_tokens = tokenize(raw_output)

        ### Convert to desired format : simplifed -> complete form

        complete_form_tokens = []
        for token in processed_tokens:
            new_token = token
            if is_opening_tag(token) and get_tag_name(token) in allowed_labels:
                new_token = str(from_simplified(token, "auto_label"))

            if is_closing_tag(token) and get_tag_name(token) in allowed_labels:
                new_token = "</auto_label>"
            
            complete_form_tokens.append(new_token)
        return complete_form_tokens
    
    def _apply_correction(
        self,
        processed_tokens: list,
        original_chunk: list
    ) -> list:
        """
        Apply Levenshtein-based error correction to align tokens.
        
        This handles minor discrepancies between the processed output and the original
        original chunk (e.g., whitespace differences, minor token variations).
        
        Args:
            processed_tokens: Post-processed token list
            original_chunk: Reference original token list
        
        Returns:
            Corrected token list
        """
        _, operations = protected_levenshtein_distance(original_chunk, processed_tokens, lambda token: is_auto_label_tag(token))
        corrected_tokens = apply_operations_safe(processed_tokens, operations, lambda token: is_auto_label_tag(token))
        return corrected_tokens
    
    def process_with_details(
        self,
        raw_llm_output: str,
        original_chunk: list,
        cleaned_chunk: list,
        label_config: dict,
        allowed_labels: Optional[List[str]] = None,
        cot: bool = False
    ) -> dict:
        """
        Process output and return detailed diagnostics.
        
        Useful for debugging and analysis.
        
        Returns:
            Dict with keys:
            - 'processed_tokens': Final processed tokens
            - 'verification': VerificationResult
            - 'passed': Boolean
            - 'output_text': Decoded output as string
            - 'error_type': Error type if failed, else None
            - 'error_details': Error details if failed, else None
        """
        processed_tokens, verification = self.process(
            raw_llm_output=raw_llm_output,
            original_chunk=original_chunk,
            allowed_labels=allowed_labels,
        )
        
        return {
            'processed_tokens': processed_tokens,
            'verification': verification,
            'passed': verification.passed,
            'output_text': decode(processed_tokens),
            'error_type': verification.error_type if not verification.passed else None,
            'error_details': verification.details if not verification.passed else None
        }
