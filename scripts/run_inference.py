import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Import components
from components.data_loader import LeaseDataLoader
from components.llm_client import LLMClient
from components.output_builder import OutputBuilder
from components.prompt_loader import PromptLoader
from dotenv import load_dotenv

"""
Field Extraction Inference Script

This script runs LLM-based inference on lease documents using optimized prompts
to extract structured field data.

Usage:
    python run_inference.py \\
        --prompts prompts/final_export \\
        --data data/interim \\
        --output data/predictions \\
        --config configs/inference_config.yaml
"""


class InferenceRunner:
    """Main runner for field extraction inference."""

    def __init__(self, config: dict):
        """Initialize the inference runner.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Setup logging
        self._setup_logging()

        # Load OpenAI API key
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        # Initialize components
        llm_config = config.get("llm", {})
        retry_config = config.get("retry", {})

        self.llm_client = LLMClient(
            api_key=api_key,
            model=llm_config.get("model", "gpt-4o-mini"),
            temperature=llm_config.get("temperature", 0.0),
            max_tokens=llm_config.get("max_tokens", 4096),
            timeout=llm_config.get("timeout", 60),
            max_retries=retry_config.get("max_attempts", 3),
            initial_delay=retry_config.get("initial_delay", 1.0),
            backoff_factor=retry_config.get("backoff_factor", 2.0),
            max_delay=retry_config.get("max_delay", 10.0),
        )

        paths = config.get("paths", {})
        doc_config = config.get("documents", {})

        self.prompt_loader = PromptLoader(prompts_dir=Path(paths.get("prompts_dir", "prompts/final_export")))

        self.data_loader = LeaseDataLoader(
            data_dir=Path(paths.get("data_dir", "data/interim")),
            vlm_suffix=doc_config.get("vlm_file_suffix", "_vlm.txt"),
            meta_suffix=doc_config.get("meta_file_suffix", "_meta.json"),
            file_separator=doc_config.get("file_separator", "\n\n" + "=" * 80 + "\n\n"),
            include_filename=doc_config.get("include_filename", True),
        )

        self.output_builder = OutputBuilder()

        self.output_dir = Path(paths.get("output_dir", "data/predictions"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save a copy of fields_config.yaml in output directory for reference
        self._save_fields_config_copy()

        # Load system prompt if specified
        system_prompt_file = paths.get("system_prompt_file")
        self.system_prompt = None
        if system_prompt_file:
            self.system_prompt = self._load_system_prompt(Path(system_prompt_file))

        # Processing config
        proc_config = config.get("processing", {})
        self.skip_existing = proc_config.get("skip_existing", True)
        self.save_intermediate = proc_config.get("save_intermediate", True)

        logging.info("Inference runner initialized successfully")

    def _setup_logging(self):
        """Setup logging based on configuration."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))

        # Create log file path with timestamp
        log_file_template = log_config.get("log_file", "logs/inference_{timestamp}.log")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_file_template.format(timestamp=timestamp)

        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure logging
        handlers = [logging.FileHandler(log_file)]

        if log_config.get("console_output", True):
            handlers.append(logging.StreamHandler())

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )

        logging.info(f"Logging initialized. Log file: {log_file}")

    def _save_fields_config_copy(self) -> None:
        """Save a copy of fields_config.yaml to the output directory for reference."""
        try:
            fields_config_path = Path("configs/fields_config.yaml")
            if fields_config_path.exists():
                output_config_path = self.output_dir / "fields_config.yaml"
                shutil.copy2(fields_config_path, output_config_path)
                logging.info(f"Saved fields_config.yaml copy to {output_config_path}")
            else:
                logging.warning(f"fields_config.yaml not found at {fields_config_path}")
        except Exception as e:
            logging.warning(f"Could not save fields_config.yaml copy: {e}")

    def _load_system_prompt(self, path: Path) -> str | None:
        """Load system prompt from file.

        Args:
            path: Path to system prompt file

        Returns:
            System prompt string, or None if loading failed
        """
        try:
            with open(path, encoding="utf-8") as f:
                prompt = f.read().strip()
            logging.info(f"Loaded system prompt from {path}")
            return prompt
        except Exception as e:
            logging.error(f"Error loading system prompt from {path}: {e}")
            return None

    def process_lease(
        self,
        lease_folder: Path,
        prompts_data: dict,
    ) -> bool:
        """Process a single lease folder.

        Args:
            lease_folder: Path to the lease subfolder
            prompts_data: Pre-loaded prompts data

        Returns:
            True if successful, False otherwise
        """
        lease_name = lease_folder.name
        logging.info(f"{'=' * 80}")
        logging.info(f"Processing lease: {lease_name}")
        logging.info(f"{'=' * 80}")

        # Check if output already exists
        output_path = self.output_dir / lease_name / "predicted_fields.json"
        if self.skip_existing and output_path.exists():
            logging.info(f"Output already exists for {lease_name}, skipping")
            return True

        # Load lease data
        logging.info("Loading lease documents...")
        lease_data = self.data_loader.load_lease_data(lease_folder)

        if lease_data is None:
            logging.error(f"Failed to load lease data for {lease_name}")
            return False

        lease_text, ground_truth = lease_data
        logging.info(f"Loaded lease text: {len(lease_text)} characters")

        if ground_truth:
            logging.info("Ground truth data available")

        # Build inference prompts for all fields
        logging.info("Building inference prompts...")
        inference_prompts = self.prompt_loader.build_all_inference_prompts(
            document_text=lease_text,
            prompts_data=prompts_data,
        )
        logging.info(f"Built {len(inference_prompts)} inference prompts")

        # Run inference for all fields
        logging.info("Running LLM inference...")
        field_responses = self.llm_client.batch_call(
            prompts=inference_prompts,
            system_prompt=self.system_prompt,
        )

        # Build structured output
        logging.info("Building structured output...")
        output_data = self.output_builder.build_output_structure(
            field_responses=field_responses,
            prompts_data=prompts_data,
        )

        # Add metadata
        output_data["_metadata"] = {
            "lease_name": lease_name,
            "lease_folder": str(lease_folder),
            "inference_timestamp": datetime.now().isoformat(),
            "model": self.config.get("llm", {}).get("model"),
            "num_fields_processed": len(field_responses),
            "num_fields_successful": sum(1 for v in field_responses.values() if v is not None),
        }

        # Save output
        logging.info(f"Saving output to {output_path}")
        success = self.output_builder.save_output(output_data, output_path)

        if success:
            logging.info(f"Successfully processed {lease_name}")
        else:
            logging.error(f"Failed to save output for {lease_name}")

        return success

    def run(self):
        """Run inference on all lease folders."""
        logging.info("=" * 80)
        logging.info("STARTING FIELD EXTRACTION INFERENCE")
        logging.info("=" * 80)

        # Load all prompts once
        logging.info("Loading prompts...")
        prompts_data = self.prompt_loader.load_all_prompts()
        logging.info(f"Loaded {len(prompts_data)} prompts")

        # Get all lease folders
        lease_folders = self.data_loader.list_lease_folders()
        total_leases = len(lease_folders)
        logging.info(f"Found {total_leases} lease folders to process")

        # Process each lease
        successful = 0
        failed = 0

        for idx, lease_folder in enumerate(lease_folders, 1):
            logging.info(f"\nProcessing lease {idx}/{total_leases}")

            try:
                success = self.process_lease(lease_folder, prompts_data)
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logging.error(f"Unexpected error processing {lease_folder.name}: {e}", exc_info=True)
                failed += 1

        # Print summary
        logging.info("=" * 80)
        logging.info("INFERENCE COMPLETE")
        logging.info("=" * 80)
        logging.info(f"Total leases:     {total_leases}")
        logging.info(f"✓ Successful:     {successful}")
        logging.info(f"✗ Failed:         {failed}")
        logging.info(f"Output directory: {self.output_dir.absolute()}")
        logging.info("=" * 80)

        return failed == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run field extraction inference on lease documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python run_inference.py

  # Specify custom paths
  python run_inference.py \\
      --prompts prompts/final_export \\
      --data data/interim \\
      --output data/predictions

  # Use custom config file
  python run_inference.py \\
      --config configs/custom_inference.yaml
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/inference_config.yaml"),
        help="Path to inference configuration file",
    )

    parser.add_argument("--prompts", type=Path, help="Override prompts directory from config")

    parser.add_argument("--data", type=Path, help="Override data directory from config")

    parser.add_argument("--output", type=Path, help="Override output directory from config")

    parser.add_argument("--system-prompt", type=Path, help="Override system prompt file from config")

    args = parser.parse_args()

    # Load config
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override config with command-line arguments
    if args.prompts:
        config.setdefault("paths", {})["prompts_dir"] = str(args.prompts)

    if args.data:
        config.setdefault("paths", {})["data_dir"] = str(args.data)

    if args.output:
        config.setdefault("paths", {})["output_dir"] = str(args.output)

    if args.system_prompt:
        config.setdefault("paths", {})["system_prompt_file"] = str(args.system_prompt)

    # Run inference
    try:
        runner = InferenceRunner(config)
        success = runner.run()
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
