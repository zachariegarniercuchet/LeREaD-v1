# Output Control Module

A unified, reusable module for managing LLM output validation, correction, and error recovery.

## Overview

This module orchestrates the complete pipeline for controlling and validating LLM-generated annotated text:

1. **Post-Processing**: Tokenization, format conversion, error correction
2. **Verification**: Multi-level checks (hallucination, consistency, label scheme, nesting)
3. **Fallback Mechanism**: Automatic retry with corrected prompts when verification fails
4. **Tracking**: Complete history and statistics of all processing attempts

## Architecture

```
output_control/
├── __init__.py           # Public API
├── controller.py         # Main orchestrator (OutputController)
├── processor.py          # Post-processing & verification (OutputProcessor)
├── fallback.py           # Fallback retry logic (FallbackHandler)
└── tracking.py           # History & statistics (OutputHistory, ProcessingRecord)
```

## Quick Start

### Basic Usage: Process a Single Output

```python
from src.output_control import OutputController

controller = OutputController()

# Process raw LLM output
tokens, status = controller.process_output(
    raw_llm_output=llm_generated_text,
    original_chunk=original_tokens,
    label_config={'use_simplified': False},
    model=model,
    enable_fallback=True,
    chunk_idx=0
)

if status['passed']:
    print("✓ Successfully processed!")
else:
    print(f"✗ Failed: {status['error_type']}")
    print(f"  Details: {status['error_details']}")
```

### Processing Multiple Chunks

```python
controller = OutputController()

for idx, chunk in enumerate(chunks):
    tokens, status = controller.process_output(
        raw_llm_output=raw_outputs[idx],
        original_chunk=chunk,
        label_config=config,
        model=model,
        enable_fallback=True,
        chunk_idx=idx
    )

# Print summary
controller.print_summary()

# Save history
controller.save_history(
    output_dir='./results',
    filename='processing_run',
    include_raw_output=False
)
```

## Components

### OutputController (Main Orchestrator)

**Purpose**: Single entry point for the complete output control pipeline.

**Key Methods**:
- `process_output()`: Main method, orchestrates full pipeline
- `save_history()`: Save processing history to JSON
- `print_summary()`: Print summary statistics
- `get_summary()`: Get summary as dict
- `get_failed_records()`: Get list of failed records

**Example**:
```python
controller = OutputController()

tokens, result = controller.process_output(
    raw_llm_output=output,
    original_chunk=chunk,
    label_config=config,
    model=model,
    enable_fallback=True,
    chunk_idx=0
)

print(result['status'])  # "Success" or error description
print(result['error_type'])  # None or "hallucination", "consistency", etc.
print(result['fallback_used'])  # Whether fallback was triggered
```

**Status Dict Keys**:
- `passed`: bool - Whether verification passed
- `status`: str - Human-readable status
- `error_type`: str or None - Type of error if failed
- `error_details`: str or None - Detailed error message
- `fallback_used`: bool - Whether fallback mechanism was used
- `fallback_passed`: bool or None - Result of fallback if used
- `output_text`: str - Decoded output tokens

### OutputProcessor

**Purpose**: Post-processes LLM output and verifies it.

**Key Methods**:
- `process()`: Core processing pipeline
- `process_with_details()`: Same but returns diagnostic dict

**Pipeline**:
1. **Post-process**: Tokenize, extract markers, convert format
2. **Correct**: Apply Levenshtein alignment for minor discrepancies
3. **Verify**: Check hallucination, consistency, label scheme, nesting

**Example**:
```python
from src.output_control import OutputProcessor

processor = OutputProcessor()

tokens, verification = processor.process(
    raw_llm_output=output,
    original_chunk=chunk,
    cleaned_chunk=cleaned,
    label_config=config,
    allowed_labels=['legislation', 'decision', ...],
    cot=False
)

if verification.passed:
    # Use tokens
    pass
else:
    print(f"Error: {verification.error_type}")
    print(f"Details: {verification.details}")
```

### FallbackHandler

**Purpose**: Handles verification failures with automatic retry.

**Key Methods**:
- `attempt_correction()`: Retry with fallback prompt
- `handle_failure()`: Smart failure handling (nesting vs. other errors)

**Strategy**:
- **Nesting errors**: Return processed tokens without fallback (minor structural issue)
- **Other errors**: Attempt fallback with error context for LLM to correct

**Example**:
```python
from src.output_control import OutputProcessor, FallbackHandler

processor = OutputProcessor()
fallback = FallbackHandler(processor)

tokens, verification = processor.process(...)

if not verification.passed:
    corrected, result = fallback.handle_failure(
        model=model,
        raw_output=output,
        original_chunk=chunk,
        cleaned_chunk=cleaned,
        original_text=text,
        initial_verification=verification,
        label_config=config,
        fallback_prompt_path='prompts/fallback.txt',
        cot=False
    )
```

### OutputHistory / ProcessingRecord

**Purpose**: Track and analyze processing history.

**Key Methods**:
- `add()`: Add a processing record
- `summary()`: Get summary statistics
- `save()`: Save to JSON
- `load()`: Load from JSON
- `print_summary()`: Print to console
- `get_failed_records()`: Get failed attempts
- `get_by_error_type()`: Filter by error type

**Example**:
```python
from src.output_control import OutputHistory

history = OutputHistory()

history.add(
    chunk_idx=0,
    status="Success",
    raw_output="...",
    fallback_used=False
)

history.add(
    chunk_idx=1,
    status="Success (after fallback)",
    raw_output="...",
    error_type="hallucination",
    fallback_used=True,
    fallback_passed=True
)

summary = history.summary()
print(f"Success rate: {summary['successful']}/{summary['total']}")

history.save('./results', 'run_001')
```

## Configuration

### Label Config

```python
label_config = {
    'use_simplified': False,      # Whether to convert simplified format
    'keep_attributes': ['labelname'],  # Which attributes to preserve
    'switch_type': False,         # Whether to switch label types
}
```

### Fallback Prompt File

The fallback prompt should guide the LLM to correct specific errors. Example format:

```
You are a linguistic annotation expert. Your task is to correct annotation errors in legal text.

When correcting annotations:
1. Preserve all original text content exactly
2. Fix only the specific error type indicated
3. Maintain proper tag nesting
4. Return only the corrected text

Important: Do not modify the text, only fix the annotations.
```

## Return Values

### process_output() Returns

```python
processed_tokens: list  # Corrected token list

status_dict: {
    'passed': bool,
    'status': str,                    # "Success", "Hallucination Fail", etc.
    'error_type': str or None,        # "hallucination", "consistency", etc.
    'error_details': str or None,     # Detailed error message
    'fallback_used': bool,            # Whether fallback was attempted
    'fallback_passed': bool or None,  # Result of fallback if used
    'output_text': str,               # Decoded output
    'note': str or None               # Additional notes (e.g., "Nesting error handled")
}
```

### Processing Summary

```python
{
    'total': int,                          # Total records
    'successful': int,                     # Successful records
    'failed': int,                         # Failed records
    'errors_by_type': {'error_type': count},  # Error breakdown
    'fallback_attempts': int,              # Number of fallback attempts
    'fallback_successful': int,            # Successful fallbacks
    'fallback_failed': int                 # Failed fallbacks
}
```

## Error Types

The module recognizes and handles these error types:

- **hallucination**: LLM modified original text content
- **consistency**: Tag nesting is unbalanced or broken
- **label_scheme**: Invalid label names or attributes
- **nesting**: Structural labels not properly nested
- **generation_error**: LLM generation failed

## Integration Example

Here's how to integrate with your main processing pipeline:

```python
from src.output_control import OutputController

class ChunkProcessor:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.output_controller = OutputController()
    
    def process_chunk(self, chunk, idx):
        """Process a single chunk through LLM."""
        
        # Generate LLM output
        llm_output = self.model.generate(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt
        )
        
        # Control and verify output
        tokens, status = self.output_controller.process_output(
            raw_llm_output=llm_output,
            original_chunk=chunk,
            label_config=self.config['label_config'],
            model=self.model,
            enable_fallback=True,
            chunk_idx=idx
        )
        
        if status['passed']:
            return tokens, "Success"
        else:
            return tokens, status['status']
    
    def process_all(self, chunks):
        """Process all chunks."""
        results = []
        for idx, chunk in enumerate(chunks):
            tokens, status = self.process_chunk(chunk, idx)
            results.append((tokens, status))
        
        # Print final summary
        self.output_controller.print_summary()
        
        return results
```

## Performance Considerations

- **History Size**: Large history objects accumulate memory. Call `save()` and create new instances periodically.
- **Fallback Cost**: Fallback attempts double the LLM calls. Set `enable_fallback=False` if not needed.
- **Raw Output Storage**: Include `include_raw_output=False` when saving to reduce file size.

## Troubleshooting

### Verification keeps failing after fallback

- Check the fallback prompt is properly formatted and instructive
- Verify the LLM model can understand the error description
- Enable `process_with_details()` to debug the exact failure point

### Nesting errors not being handled

- Use `enable_nesting_fallback=True` (default) to return nesting errors as-is
- Nesting errors are structural; content is usually correct

### Memory issues with large histories

- Call `save()` periodically and create new `OutputController` instances
- Use `include_raw_output=False` to reduce memory footprint

## Testing

Test individual components:

```python
from src.output_control import OutputProcessor, FallbackHandler

# Test processor
processor = OutputProcessor()
tokens, result = processor.process(...)

# Test fallback
fallback = FallbackHandler(processor)
tokens, status = fallback.handle_failure(...)

# Test history
from src.output_control import OutputHistory
history = OutputHistory()
history.add(...)
summary = history.summary()
```

---

**Created**: 2026-06-01  
**Module Version**: 1.0  
**Purpose**: Unified, reusable LLM output control pipeline
