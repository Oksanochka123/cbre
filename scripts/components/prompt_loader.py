"""Prompt loader for field extraction inference."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptLoader:
    """Loader for optimized field extraction prompts."""

    def __init__(self, prompts_dir: Path):
        """Initialize the prompt loader.

        Args:
            prompts_dir: Directory containing exported prompt JSON files
        """
        self.prompts_dir = Path(prompts_dir)

        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")

        logger.info(f"Initialized prompt loader for directory: {self.prompts_dir}")

    def load_prompt(self, field_name: str) -> dict | None:
        """Load prompt data for a specific field.

        Args:
            field_name: Name of the field

        Returns:
            Prompt data dictionary, or None if not found
        """
        prompt_file = self.prompts_dir / f"{field_name}.json"

        if not prompt_file.exists():
            logger.debug(f"Prompt file not found for field: {field_name}")
            return None

        try:
            with open(prompt_file, encoding="utf-8") as f:
                data = json.load(f)

            logger.debug(f"Loaded prompt for field: {field_name}")
            return data

        except Exception as e:
            logger.error(f"Error loading prompt for {field_name}: {e}")
            return None

    def load_all_prompts(self) -> dict[str, dict]:
        """Load all available prompts.

        Returns:
            Dictionary mapping field names to prompt data
        """
        prompts = {}

        # Find all JSON files in the prompts directory
        json_files = list(self.prompts_dir.glob("*.json"))

        logger.info(f"Found {len(json_files)} prompt files")

        for json_file in json_files:
            field_name = json_file.stem  # Filename without extension

            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                prompts[field_name] = data
                logger.debug(f"Loaded prompt for field: {field_name}")

            except Exception as e:
                logger.error(f"Error loading prompt from {json_file.name}: {e}")
                continue

        logger.info(f"Successfully loaded {len(prompts)} prompts")
        return prompts

    def build_inference_prompt(
        self,
        field_name: str,
        document_text: str,
        prompt_data: dict | None = None,
    ) -> dict | None:
        """Build the complete inference prompt for a field using DSPy ChatAdapter format.

        Uses the pre-computed final_prompt from the exported data, which follows
        DSPy ChatAdapter formatting with separate system and user messages.

        Args:
            field_name: Name of the field
            document_text: The lease document text
            prompt_data: Optional pre-loaded prompt data (will load if not provided)

        Returns:
            Dictionary with 'system' and 'user' messages, or None if prompt not available
        """
        if prompt_data is None:
            prompt_data = self.load_prompt(field_name)

        if prompt_data is None:
            logger.warning(f"Cannot build prompt for {field_name}: prompt data not found")
            return None

        # Check for final_prompt
        final_prompt = prompt_data.get("final_prompt")
        lease_document_text = "Lease Document Text:`````\n\n" + document_text + "`````"
        if final_prompt is not None:
            # Handle new string format (single combined prompt)
            if isinstance(final_prompt, str):
                # Replace placeholder with actual document text
                user_message = final_prompt.replace("{document_text}", lease_document_text)

                logger.debug(f"Built inference prompt for {field_name}, " f"length: {len(user_message)} chars")

                return {
                    "system": None,
                    "user": user_message,
                }

            # Handle legacy dict format with system/user_template
            elif isinstance(final_prompt, dict):
                system_msg = final_prompt.get("system")
                user_tmpl = final_prompt.get("user_template")

                if system_msg and user_tmpl:
                    # Use the DSPy ChatAdapter formatted prompt
                    # Replace placeholder with actual document text
                    user_message = user_tmpl.replace("{document_text}", document_text)

                    logger.debug(
                        f"Built DSPy-formatted inference prompt for {field_name}, "
                        f"system: {len(system_msg)} chars, user: {len(user_message)} chars"
                    )

                    return {
                        "system": system_msg,
                        "user": user_message,
                    }

        # Fallback to legacy format for backward compatibility
        instructions = prompt_data.get("instructions", "")
        if not instructions:
            logger.warning(f"No instructions found in prompt for {field_name}")
            return None

        # Build legacy format as single user message
        user_message = f"""# TASK INSTRUCTIONS

{instructions}

# DOCUMENT TEXT

{document_text}

# YOUR RESPONSE

Please extract the requested information from the document above.
"""

        logger.debug(f"Built legacy inference prompt for {field_name}, " f"length: {len(user_message)} chars")

        return {
            "system": None,
            "user": user_message,
        }

    def build_all_inference_prompts(
        self,
        document_text: str,
        prompts_data: dict[str, dict] | None = None,
    ) -> dict[str, dict]:
        """Build inference prompts for all available fields.

        Args:
            document_text: The lease document text
            prompts_data: Optional pre-loaded prompts data (will load if not provided)

        Returns:
            Dictionary mapping field names to prompt dicts with 'system' and 'user' keys
        """
        if prompts_data is None:
            prompts_data = self.load_all_prompts()

        inference_prompts = {}

        for field_name, prompt_data in prompts_data.items():
            prompt = self.build_inference_prompt(
                field_name=field_name,
                document_text=document_text,
                prompt_data=prompt_data,
            )

            if prompt:
                inference_prompts[field_name] = prompt

        logger.info(f"Built {len(inference_prompts)} inference prompts " f"from {len(prompts_data)} available prompts")

        return inference_prompts
