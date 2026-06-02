"""
Output Control Module: Unified management of LLM output validation and correction.

This module provides a clean, reusable interface for:
- Post-processing and verification of LLM output
- Fallback mechanisms for handling verification failures
- Tracking processing history and statistics

Main Components:
- OutputController: Main orchestrator (recommended entry point)
- OutputProcessor: Post-processing and verification
- FallbackHandler: Fallback retry mechanism
- OutputHistory: History tracking and statistics

Usage Examples:
    # Simple usage with OutputController
    from src.output_control import OutputController
    
    controller = OutputController()
    tokens, status = controller.process_output(
        raw_llm_output=llm_output,
        original_chunk=chunk,
        label_config=config,
        model=model,
        enable_fallback=True
    )
    
    if status['passed']:
        print("Success!")
    
    # Advanced: custom components
    from src.output_control import OutputProcessor, FallbackHandler
    
    processor = OutputProcessor()
    fallback = FallbackHandler(processor)
    
    tokens, verification = processor.process(...)
    if not verification.passed:
        tokens, status = fallback.handle_failure(...)
"""

from .processor import OutputProcessor
from .fallback import FallbackHandler
from .tracking import OutputHistory, ProcessingRecord

__all__ = [
    'OutputProcessor',
    'FallbackHandler',
    'OutputHistory',
    'ProcessingRecord'
]
