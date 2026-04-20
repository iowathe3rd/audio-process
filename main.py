import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from app.config import PipelineConfig
from app.pipeline.orchestrator import PipelineOrchestrator

load_dotenv()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean OOP audio processing pipeline",
    )
    parser.add_argument(
        "--input",
        default="audio.wav",
        help="Input audio file",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory for artifacts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run all stages (invalidate cache)",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = PipelineConfig(
        input_path=Path(args.input),
        artifacts_root=Path(args.artifacts_dir),
        force=args.force,
        hf_token=os.getenv("HF_TOKEN", ""),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
    )

    orchestrator = PipelineOrchestrator(config)
    try:
        result = orchestrator.run()
        print(f"Pipeline finished successfully. Results at: {result['artifacts_dir']}")
    except Exception:
        logging.exception("Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
