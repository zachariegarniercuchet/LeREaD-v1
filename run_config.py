# run_config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal

@dataclass
class RunConfig:
    """Experiment-level configuration resolved once at launch."""
    
    # Mode
    mode: Literal["single", "split"]           # single file vs entire split
    
    # Single-mode only
    filename: Optional[str] = None             # e.g. "1989CanLII1415ONCA"
    
    # Split-mode only  
    split: Optional[str] = None               # "train" | "dev" | "test"
    
    # Shared
    method: str = "AIO"
    chunker: str = "paragraph"
    fewshot_method: str = "greedy"
    fewshot_examples: int = 6
    model: str = "gpt-5.2"
    temperature: float = 1.0
    disable_fallback: bool = False
    with_timestamp: bool = False
    output_dir: Path = Path("./output")
    prompt_type: str = "long"

    def output_root(self) -> Path:
        """Resolve the top-level output folder for this run."""
        if self.mode == "split":
            # e.g. output/test_AIO_fs6_greedy_paragraph/
            folder = f"{self.split}_{self.model}_{self.method}_fs{self.fewshot_examples}_{self.fewshot_method}_{self.chunker}_{self.prompt_type}"
        else:
            # e.g. output/single_1989CanLII1415ONCA_AIO_.../
            folder = f"single_{self.filename}_{self.model}_{self.method}_fs{self.fewshot_examples}_{self.fewshot_method}_{self.chunker}_{self.prompt_type}"
        
        if self.with_timestamp:
            from datetime import datetime
            folder += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        root = self.output_dir / folder
        root.mkdir(parents=True, exist_ok=True)
        return root

    def get_filenames(self, data_dir: Path) -> List[str]:
        """Return the list of filenames to process."""
        if self.mode == "single":
            return [self.filename]
        split_dir = data_dir / "original" / self.split
        return sorted(p.stem for p in split_dir.glob("*.html"))