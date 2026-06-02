"""
Output Tracking: Manages processing history, statistics, and error logging.
Tracks all LLM output processing attempts for analysis and debugging.
"""

import os
import json
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ProcessingRecord:
    """A single record of chunk/output processing."""
    chunk_idx: int
    status: str  # "Success", "Hallucination Fail", "Consistency Fail", etc.
    raw_output: str
    error_type: Optional[str] = None
    error_details: Optional[str] = None
    fallback_used: bool = False
    fallback_passed: Optional[bool] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class OutputHistory:
    """
    Tracks processing history for all chunks/outputs.
    
    Provides:
    - Recording individual processing attempts
    - Summary statistics
    - Saving/loading history from JSON
    - Analysis and debugging info
    """
    
    def __init__(self):
        """Initialize empty history."""
        self.entries: List[ProcessingRecord] = []
    
    def add(
        self,
        chunk_idx: int,
        status: str,
        raw_output: str,
        error_type: Optional[str] = None,
        error_details: Optional[str] = None,
        fallback_used: bool = False,
        fallback_passed: Optional[bool] = None
    ) -> None:
        """
        Add a processing record to history.
        
        Args:
            chunk_idx: Index of the chunk processed
            status: Processing status (e.g., "Success", "Hallucination Fail")
            raw_output: The processed output (as string)
            error_type: Type of error if failed (hallucination, consistency, etc.)
            error_details: Detailed error message
            fallback_used: Whether fallback mechanism was used
            fallback_passed: Whether fallback attempt passed (if used)
        """
        record = ProcessingRecord(
            chunk_idx=chunk_idx,
            status=status,
            raw_output=raw_output,
            error_type=error_type,
            error_details=error_details,
            fallback_used=fallback_used,
            fallback_passed=fallback_passed
        )
        self.entries.append(record)
    
    def summary(self) -> Dict:
        """
        Get summary statistics of processing history.
        
        Returns:
            Dict with keys:
            - 'total': total records
            - 'successful': count of successful records
            - 'failed': count of failed records
            - 'errors_by_type': dict of error type → count
            - 'fallback_attempts': count of fallback attempts
            - 'fallback_successful': count of successful fallbacks
        """
        statuses = [entry.status for entry in self.entries]
        errors = [entry.error_type for entry in self.entries if entry.error_type]
        fallbacks = [entry for entry in self.entries if entry.fallback_used]
        
        error_counts = {}
        for error in errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        
        successful_fallbacks = sum(1 for fb in fallbacks if fb.fallback_passed)
        
        return {
            'total': len(self.entries),
            'successful': len([s for s in statuses if s == "Success"]),
            'failed': len([s for s in statuses if s != "Success"]),
            'errors_by_type': error_counts,
            'fallback_attempts': len(fallbacks),
            'fallback_successful': successful_fallbacks,
            'fallback_failed': len(fallbacks) - successful_fallbacks
        }
    
    def get_failed_records(self) -> List[ProcessingRecord]:
        """Get all failed processing records."""
        return [e for e in self.entries if e.status != "Success"]
    
    def get_by_error_type(self, error_type: str) -> List[ProcessingRecord]:
        """Get all records with a specific error type."""
        return [e for e in self.entries if e.error_type == error_type]
    
    def save(
        self,
        output_dir: str,
        filename: str,
        include_raw_output: bool = False
    ) -> str:
        """
        Save processing history to JSON file.
        
        Args:
            output_dir: Directory to save to
            filename: Base filename (will be prefixed with 'history_' and suffixed with '.json')
            include_raw_output: Whether to include raw LLM output in saved history
        
        Returns:
            Path to saved file
        
        Raises:
            OSError: If directory doesn't exist or can't write file
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        json_path = os.path.join(output_dir, f"history_{filename}.json")
        
        # Prepare data for serialization
        data = []
        for entry in self.entries:
            entry_dict = asdict(entry)
            if not include_raw_output:
                # Truncate or remove raw output to reduce file size
                entry_dict['raw_output'] = f"[Output truncated - {len(entry.raw_output)} chars]"
            data.append(entry_dict)
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"   ✓ Processing history saved to: {json_path}")
            return json_path
        except Exception as e:
            print(f"   ✗ Error saving history: {e}")
            raise
    
    def load(self, json_path: str) -> None:
        """
        Load processing history from JSON file.
        
        Args:
            json_path: Path to history JSON file
        
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.entries = []
        for item in data:
            # Handle timestamp format
            timestamp = item.get('timestamp')
            record = ProcessingRecord(
                chunk_idx=item['chunk_idx'],
                status=item['status'],
                raw_output=item['raw_output'],
                error_type=item.get('error_type'),
                error_details=item.get('error_details'),
                fallback_used=item.get('fallback_used', False),
                fallback_passed=item.get('fallback_passed'),
                timestamp=timestamp
            )
            self.entries.append(record)
    
    def print_summary(self) -> None:
        """Print summary statistics to console."""
        summary = self.summary()
        
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total processed:          {summary['total']}")
        print(f"Successful:               {summary['successful']}")
        print(f"Failed:                   {summary['failed']}")
        
        if summary['errors_by_type']:
            print(f"\nErrors by type:")
            for error_type, count in summary['errors_by_type'].items():
                print(f"  - {error_type}: {count}")
        
        print(f"\nFallback attempts:        {summary['fallback_attempts']}")
        print(f"  - Successful:           {summary['fallback_successful']}")
        print(f"  - Failed:               {summary['fallback_failed']}")
        print("="*60 + "\n")
    
    def __len__(self) -> int:
        """Return number of records."""
        return len(self.entries)
    
    def __repr__(self) -> str:
        return f"OutputHistory({len(self.entries)} records)"
