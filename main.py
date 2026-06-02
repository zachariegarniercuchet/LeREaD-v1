"""Main processing script for document-level annotation extraction."""

import sys
from pathlib import Path
import json
import argparse
from typing import List, Optional

# Setup project path
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, FEWSHOT_CACHE_DIR, PROMPT_DIR, LABEL_SCHEME
from src.fewshot.extractor import LabelTransformConfig, _prepare_label_tokens, _parse_parent_annotations
from src.tokenizer_utils import tokenize, decode
from src import clean_tokens
from src.chunkers.cache import cache_exists, load_cache
from src.chunkers import ChunkerFactory
from src.models import get_message, AssistantFactory
from src.output_control.processor import OutputProcessor
from src.output_control.fallback import FallbackHandler
from src.post_processing import chunks_to_html


def load_fewshot_examples(fewshot_method: str, nb_examples: int, spans_in_context: bool = True) -> List[tuple]:
    """Load and prepare few-shot examples."""
    with open(FEWSHOT_CACHE_DIR / f"examples_{fewshot_method}.json", "r", encoding="utf-8") as f:
        fewshot_file_content = json.load(f)
    
    fewshot_examples = [(example["example"]["input"], example["example"]["output"]) 
                        for example in fewshot_file_content["examples"]]
    
    allowed_labels = ["decision", "legislation", "secondary sources"]
    label_config = LabelTransformConfig(
        use_simplified=True,
        switch_type=True,
        keep_labels=allowed_labels,
        keep_attributes=["labelname"]
    )
    
    # Transform the output in its simplified form
    final_fewshot = []
    total_output_text = ""
    
    for example in fewshot_examples:
        _, output = example
        output_tokens = tokenize(output)
        transformed_output_tokens = _prepare_label_tokens(output_tokens, label_config)
        
        if spans_in_context:
            final_fewshot.append((example[0], decode(transformed_output_tokens)))
        else:
            total_output_text += "|||" + decode(transformed_output_tokens)
    
    if not spans_in_context:
        parents_dict = _parse_parent_annotations(total_output_text)
        for parent_name, annotations in parents_dict.items():
            for annotation in annotations:
                final_fewshot.append((decode(clean_tokens(tokenize(annotation))), annotation))
    
    return final_fewshot[:nb_examples], allowed_labels


def load_html_document(filename: str, split: str) -> str:
    """Load HTML content from file."""
    filepath = Path(DATA_DIR) / "original" / split / f"{filename}.html"
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def get_token_chunks(html_content: str, chunker: str, split: str, filename: str):
    """Get token chunks, using cache if available."""
    if not cache_exists(chunker, split, filename):
        nlp = None
        if chunker == "sentence":
            import spacy
            nlp = spacy.load("en_core_web_trf")
            print("✅ spaCy model loaded.\n")
        
        token_chunks = ChunkerFactory.get_chunks(
            html_content, method=chunker, split=split, filename=filename, nlp=nlp
        )
    else:
        token_chunks = load_cache(chunker, split, filename)
    
    return token_chunks


def create_message(chunk_tokens, system_prompt: str, fewshot_examples: List[tuple]) -> str:
    """Create a message for the model."""
    user_input = decode(chunk_tokens)
    message = get_message(
        system_prompt=system_prompt,
        user_input=user_input,
        fewshot_examples=fewshot_examples,
        has_system_role=True
    )
    return message


def process_chunk(chunk_tokens, assistant, allowed_labels: List[str], 
                  fallback_prompt_filename: str = "fallback.txt") -> tuple:
    """Process a single chunk through the model and verification pipeline."""
    
    # Generate output from model
    message = create_message(chunk_tokens, system_prompt="", fewshot_examples=[])
    # Note: This is a simplified version; you may need to pass system_prompt and fewshot_examples
    
    generated = assistant.generate(message=message)
    
    # Process and verify output
    controller = OutputProcessor()
    fallback_handler = FallbackHandler(processor=controller)
    
    corrected_generated_tokens, status = controller.process(
        raw_llm_output=generated,
        original_chunk=chunk_tokens,
        allowed_labels=allowed_labels
    )
    
    if not status.passed:
        print("   ⚠️  Output did not pass verification. Invoking fallback mechanism...")
        corrected_generated_tokens, status = fallback_handler.handle_failure(
            assistant=assistant,
            corrected_output=corrected_generated_tokens,
            original_chunk=chunk_tokens,
            initial_status=status,
            allowed_labels=allowed_labels,
            fallback_prompt_filename=fallback_prompt_filename
        )
    
    return corrected_generated_tokens, status


def main():
    """Main processing pipeline."""
    parser = argparse.ArgumentParser(description="Process a document for annotation extraction.")
    parser.add_argument("--filename", type=str, required=True, help="Filename without extension (e.g., '1989CanLII1415ONCA')")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "dev"], help="Data split")
    parser.add_argument("--chunker", type=str, default="paragraph", choices=["paragraph", "sentence"], help="Chunking method")
    parser.add_argument("--fewshot-method", type=str, default="greedy", choices=["greedy", "random"], help="Few-shot selection method")
    parser.add_argument("--fewshot-examples", type=int, default=6, help="Number of few-shot examples")
    parser.add_argument("--prompt", type=str, default="decomposed0_long.txt", help="Prompt filename")
    parser.add_argument("--model", type=str, default="gpt-5.2", help="Model name (e.g., 'gpt-5.2')")
    parser.add_argument("--temperature", type=float, default=1, help="Model temperature")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"), help="Output directory for processed HTML")
    parser.add_argument("--disable-fallback", action="store_true", help="Disable fallback mechanism")
    
    args = parser.parse_args()
    
    print("="*80)
    print("DOCUMENT-LEVEL ANNOTATION EXTRACTION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Filename: {args.filename}")
    print(f"  Split: {args.split}")
    print(f"  Chunker: {args.chunker}")
    print(f"  Few-shot method: {args.fewshot_method} ({args.fewshot_examples} examples)")
    print(f"  Prompt: {args.prompt}")
    print(f"  Model: {args.model} (temperature={args.temperature})")
    print()
    
    # =========================================================================
    # Step 1: Load HTML content
    # =========================================================================
    print("[Step 1] Loading HTML document...")
    html_content = load_html_document(args.filename, args.split)
    print(f"  ✓ Loaded {len(html_content)} characters\n")
    
    # =========================================================================
    # Step 2: Chunk the document
    # =========================================================================
    print("[Step 2] Chunking document...")
    token_chunks = get_token_chunks(html_content, args.chunker, args.split, args.filename)
    print(f"  ✓ Created {len(token_chunks)} chunks\n")
    
    # =========================================================================
    # Step 3: Load few-shot examples
    # =========================================================================
    print("[Step 3] Loading few-shot examples...")
    fewshot_examples, allowed_labels = load_fewshot_examples(
        args.fewshot_method, 
        args.fewshot_examples
    )
    print(f"  ✓ Loaded {len(fewshot_examples)} examples")
    print(f"  ✓ Allowed labels: {', '.join(allowed_labels)}\n")
    
    # =========================================================================
    # Step 4: Load prompt
    # =========================================================================
    print("[Step 4] Loading prompt...")
    with open(PROMPT_DIR / args.prompt, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    print(f"  ✓ Loaded prompt ({len(system_prompt)} characters)\n")
    
    # =========================================================================
    # Step 5: Initialize model
    # =========================================================================
    print("[Step 5] Initializing model...")
    model_config = {
        "type": "openai",
        "model_name": args.model,
        "temperature": args.temperature,
    }
    assistant = AssistantFactory.create_from_config(model_config)
    print(f"  ✓ Model initialized\n")
    
    # =========================================================================
    # Step 6: Process all chunks
    # =========================================================================
    print("[Step 6] Processing chunks...")
    print(f"  Processing {len(token_chunks)} chunks...\n")
    
    processed_chunks = []
    fallback_count = 0
    
    for i, chunk_tokens in enumerate(token_chunks, 1):
        print(f"  [{i}/{len(token_chunks)}] Processing chunk...")
        
        # Create message for this chunk
        user_input = decode(chunk_tokens)
        message = get_message(
            system_prompt=system_prompt,
            user_input=user_input,
            fewshot_examples=fewshot_examples,
            has_system_role=True
        )
        
        # Generate output from model
        generated = assistant.generate(message=message)
        
        # Process and verify output
        controller = OutputProcessor()
        fallback_handler = FallbackHandler(processor=controller)
        
        corrected_generated_tokens, status = controller.process(
            raw_llm_output=generated,
            original_chunk=chunk_tokens,
            allowed_labels=allowed_labels
        )
        
        if not status.passed and not args.disable_fallback:
            print(f"      ⚠️  Output did not pass verification. Invoking fallback mechanism...")
            corrected_generated_tokens, status = fallback_handler.handle_failure(
                assistant=assistant,
                corrected_output=corrected_generated_tokens,
                original_chunk=chunk_tokens,
                initial_status=status,
                allowed_labels=allowed_labels,
                fallback_prompt_filename="fallback.txt"
            )
            fallback_count += 1
        
        processed_chunks.append(corrected_generated_tokens)
        
        if status.passed:
            print(f"      ✓ Chunk passed verification")
        else:
            print(f"      ⚠️  Chunk did not pass verification (status: {status.status})")
    
    print(f"\n  ✓ Processed all {len(token_chunks)} chunks")
    if fallback_count > 0:
        print(f"  ℹ️  Fallback mechanism invoked {fallback_count} time(s)\n")
    else:
        print()
    
    # =========================================================================
    # Step 7: Post-processing
    # =========================================================================
    print("[Step 7] Post-processing document...")
    output_html_content = chunks_to_html(processed_chunks, html_content)
    print(f"  ✓ Post-processing complete\n")
    
    # =========================================================================
    # Step 8: Save output
    # =========================================================================
    print("[Step 8] Saving output...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = args.output_dir / f"{args.filename}_processed.html"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(output_html_content)
    
    print(f"  ✓ Saved to {output_filename}\n")
    
    print("="*80)
    print("✓ PROCESSING COMPLETE")
    print("="*80)
    
    return output_html_content


if __name__ == "__main__":
    main()
