"""
Thai Character Recognition - Direct Inference Entry Point.
Executes direct inference on single images or entire folders from Terminal.
Loads 'best_thai_character_finetuned.pth' by default (fallback to 'best_thai_character_model_v2.pth').

Usage:
    # 1. Single image prediction:
    python inference.py --image "path/to/image.png" --top-k 3

    # 2. Batch folder prediction with CSV export:
    python inference.py --folder "path/to/folder" --output-csv "predictions.csv"
"""

from pathlib import Path
import sys

# Ensure project root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import ThaiCharacterPredictor, main, parse_args, print_prediction_table

if __name__ == "__main__":
    main()
