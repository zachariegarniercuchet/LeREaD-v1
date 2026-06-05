"""
LeREaD Evaluation Scripts

Provides flexible evaluation of LeREaD system outputs against gold-standard annotations.

FILES
=====
- evaluate.py              Main evaluation script (CLI interface)
- evaluation_config.py     Configuration dataclass for evaluation runs
- evaluation_configs.py    Example configuration presets


QUICK START
===========

1. SINGLE FILE EVALUATION
   Evaluate one system output file:
   
   $ python src/evaluation/evaluate.py single \\
       --split test \\
       --system-file output/1989CanLII1415ONCA_processed.html
   
   Requirements:
   - Must specify --split (train/test/dev) to locate ground truth
   - System file will be matched with gold file by stem name


2. BATCH FOLDER EVALUATION
   Evaluate all files in a folder:
   
   $ python src/evaluation/evaluate.py batch \\
       --batch-folder output/test_gpt-5.2_DEC_fs6_greedy_sentence/
   
   Features:
   - Auto-detects split from folder name (e.g., "test_gpt-5.2..." → split="test")
   - Handles two folder structures:
     a) Direct HTML files: evaluates all *.html files
     b) Subfolders: each subfolder contains one *final.html
   - Matches files to ground truth by stem name


3. CONFIG FILE EVALUATION
   Use a saved configuration:
   
   $ python src/evaluation/evaluate.py config \\
       --config-file src/evaluation/evaluation_configs.py \\
       --config-name test_batch_aio_config
   
   Advantages:
   - Define experiments once, run them easily
   - Share configurations with team
   - Version control experiment setups


COMMAND-LINE REFERENCE
======================

SINGLE FILE EVALUATION
----------------------
usage: evaluate.py single [-h] --split {train,test,dev} --system-file SYSTEM_FILE
                          [--data-dir DATA_DIR] [--context-chars CONTEXT_CHARS] 
                          [--verbose]

options:
  --split {train,test,dev}     Ground truth split (required)
  --system-file FILE           Path to system output HTML file (required)
  --data-dir DIR              Root data directory (default: ./data)
  --context-chars INT         Context characters for span matching (default: 200)
  --verbose                   Show detailed output


BATCH FOLDER EVALUATION
-----------------------
usage: evaluate.py batch [-h] --batch-folder BATCH_FOLDER [--split {train,test,dev}]
                         [--data-dir DATA_DIR] [--context-chars CONTEXT_CHARS]
                         [--verbose]

options:
  --batch-folder FOLDER       Path to batch output folder (required)
  --split {train,test,dev}    Split name (optional, auto-detected from folder name)
  --data-dir DIR             Root data directory (default: ./data)
  --context-chars INT        Context characters for span matching (default: 200)
  --verbose                  Show detailed output per file


CONFIG FILE EVALUATION
----------------------
usage: evaluate.py config [-h] --config-file CONFIG_FILE [--config-name CONFIG_NAME]

options:
  --config-file FILE         Path to Python file with EvaluationConfig (required)
  --config-name NAME         Variable name in config file (default: 'config')


CONFIGURATION EXAMPLES
======================

# Define a configuration in evaluation_configs.py:
from pathlib import Path
from src.evaluation.evaluation_config import EvaluationConfig

my_experiment = EvaluationConfig(
    mode="batch",
    split="test",
    batch_folder="./output/test_gpt-5.2_DEC_fs6_greedy_sentence/",
    verbose_per_file=False,
)

# Then use it:
$ python src/evaluation/evaluate.py config \\
    --config-file src/evaluation/evaluation_configs.py \\
    --config-name my_experiment


BATCH FOLDER STRUCTURES
=======================

The script automatically detects and handles two batch folder structures:

1. DIRECT HTML FILES
   Folder structure:
   output/test_gpt-5.2_DEC_fs6_greedy_sentence/
   ├── 1989CanLII1415ONCA_processed.html
   ├── 2005QCCA437_processed.html
   └── 2024NBKB203_processed.html
   
   Each .html file is matched with a gold file by stem name:
   - 1989CanLII1415ONCA_processed.html → data/annotated/{split}/1989CanLII1415ONCA.html

2. SUBFOLDER WITH FINAL.HTML
   Folder structure:
   output/test_gpt-5.2_DEC_fs6_greedy_sentence/
   ├── 1989CanLII1415ONCA_AIO_fs6_greedy_paragraph/
   │   └── 1989CanLII1415ONCA_processed_final.html
   ├── 2005QCCA437_AIO_fs6_greedy_paragraph/
   │   └── 2005QCCA437_processed_final.html
   └── 2024NBKB203_AIO_fs6_greedy_paragraph/
       └── 2024NBKB203_processed_final.html
   
   Each subfolder's *final.html is evaluated. File matching uses the
   subfolder base name (first component before underscore):
   - 1989CanLII1415ONCA_AIO_... → 1989CanLII1415ONCA.html


WORKFLOW EXAMPLES
=================

SCENARIO 1: Quick single file evaluation
$ python src/evaluation/evaluate.py single \\
    --split test \\
    --system-file output/1989CanLII1415ONCA_processed.html \\
    --verbose


SCENARIO 2: Batch evaluation with auto-detected split
$ python src/evaluation/evaluate.py batch \\
    --batch-folder output/test_gpt-5.2_DEC_fs6_greedy_sentence/


SCENARIO 3: Verbose batch evaluation (see each file's results)
$ python src/evaluation/evaluate.py batch \\
    --batch-folder output/test_gpt-5.2_DEC_fs6_greedy_sentence/ \\
    --verbose


SCENARIO 4: Save configuration, use multiple times
# In evaluation_configs.py:
my_eval_config = EvaluationConfig(
    mode="batch",
    split="test",
    batch_folder="./output/test_gpt-5.2_DEC_fs6_greedy_sentence/",
    verbose_per_file=False,
)

# Then run whenever needed:
$ python src/evaluation/evaluate.py config \\
    --config-file src/evaluation/evaluation_configs.py \\
    --config-name my_eval_config


OUTPUT
======

The evaluation produces:
- Per-label metrics: Precision, Recall, F1
- Per-document F1 scores (micro-averaged)
- Macro-averaged F1 (mean of per-document F1s)
- Summary statistics: true positives, gold count, system count
- Skipped files (if any) with error messages

Example output for batch evaluation:
============================================================
BATCH EVALUATION
============================================================
Batch folder: output/test_gpt-5.2_DEC_fs6_greedy_sentence/
Split: test
============================================================

Found 3 evaluation pair(s)

Batch evaluation — 3 file pair(s)
============================================================

[1/3] 1989CanLII1415ONCA.html
  micro F1: 45.2%

[2/3] 2005QCCA437.html
  micro F1: 52.3%

[3/3] 2024NBKB203.html
  micro F1: 48.1%

============================================================
RESULTS (all documents)
  label        tp  n_gold  n_system      P      R     F1  macro
────────────────────────────────────────────────────────────────
  legislation   42      89       87   0.48   0.47   0.48
  decision      38      78       80   0.48   0.49   0.48
  title         35      45       48   0.73   0.78   0.75
  citation      55      120      118   0.47   0.46   0.46
  ...
────────────────────────────────────────────────────────────────
  TOTAL        520     1223     1245   0.42   0.43   0.42  0.51


GROUND TRUTH LOCATION
======================

Gold-standard annotations are expected in:
  data/annotated/{split}/

Where {split} is one of: train, test, dev

Example structure:
data/
├── annotated/
│   ├── train/
│   │   ├── 1989CanLII1415ONCA.html
│   │   ├── 2005QCCA437.html
│   │   └── ...
│   ├── test/
│   │   ├── 1989CanLII1415ONCA.html
│   │   ├── 2005QCCA437.html
│   │   └── ...
│   └── dev/
│       └── ...
└── ...


ERROR HANDLING
==============

The script provides helpful error messages for common issues:

1. File not found
   ❌ Error: System file not found: output/nonexistent.html

2. Gold file not found
   ❌ Error: Gold file not found: data/annotated/test/1989CanLII1415ONCA.html
      Expected file: 1989CanLII1415ONCA.html in data/annotated/test

3. Split not provided for single mode
   ❌ Error: mode='single' requires split parameter

4. No split detected in batch folder name
   ❌ Error: Could not auto-detect split from folder name: my_output_folder/
      Please provide split parameter explicitly.

5. No files found in batch folder
   ❌ Error: No HTML files or subfolders found in batch folder


ADVANCED USAGE
==============

1. CUSTOM DATA DIRECTORY
   You can point to a different ground truth directory:
   
   $ python src/evaluation/evaluate.py batch \\
       --batch-folder output/test_gpt-5.2_DEC_fs6_greedy_sentence/ \\
       --data-dir /path/to/custom/data/


2. CUSTOM CONTEXT WINDOW
   Adjust the context characters used for span matching:
   
   $ python src/evaluation/evaluate.py single \\
       --split test \\
       --system-file output/1989CanLII1415ONCA_processed.html \\
       --context-chars 100


3. CONFIG FILES WITH IMPORTS
   Your config file can import from other modules:
   
   from pathlib import Path
   from src.evaluation.evaluation_config import EvaluationConfig
   from config import PROJECT_ROOT
   
   my_config = EvaluationConfig(
       mode="batch",
       split="test",
       batch_folder=str(PROJECT_ROOT / "output" / "test_gpt-5.2_DEC_fs6_greedy_sentence"),
       data_dir=PROJECT_ROOT / "data",
   )


EXTENDING THE SCRIPT
====================

To add custom evaluation logic, you can:

1. Subclass EvaluationConfig:
   
   class MyCustomEvalConfig(EvaluationConfig):
       custom_param: str = "default"
       
       def my_custom_method(self):
           ...

2. Extend evaluate.py with new commands:
   
   # Add new subparser for your custom mode
   custom_parser = subparsers.add_parser("custom")
   # ... add arguments ...
   
   # Add handler in main():
   elif args.command == "custom":
       # Your custom logic

3. Import and use evaluate_batch directly:
   
   from src.evaluation.evaluate import collect_html_pairs_from_batch_folder
   from src.evaluation.evaluation_l1_util import evaluate_batch
   
   pairs = collect_html_pairs_from_batch_folder(...)
   results = evaluate_batch(pairs)
"""
