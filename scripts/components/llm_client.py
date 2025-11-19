"""LLM Client with retry logic for field extraction inference."""

import logging
import time

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for making LLM API calls with retry logic."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 60,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 10.0,
    ):
        """Initialize the LLM client.

        Args:
            api_key: OpenAI API key
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay before first retry (seconds)
            backoff_factor: Exponential backoff multiplier
            max_delay: Maximum delay between retries (seconds)
        """
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay

        logger.info(f"Initialized LLM client with model: {model}")

    def call(
        self,
        prompt: str,
        system_prompt: str | None = None,
        field_name: str | None = None,
    ) -> str | None:
        """Make an LLM API call with retry logic.

        Args:
            prompt: The user prompt/task
            system_prompt: Optional system prompt
            field_name: Optional field name for logging

        Returns:
            The LLM response text, or None if all retries failed
        """
        messages = []

        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add user message
        messages.append({"role": "user", "content": prompt})

        # Attempt the call with retries
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{self.max_retries} for field: {field_name or 'unknown'}")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                # Extract the response text
                result = response.choices[0].message.content

                logger.debug(f"Successfully received response for field: {field_name or 'unknown'}")
                return result

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for field " f"'{field_name or 'unknown'}': {e}"
                )

                # If this was the last attempt, give up
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"All {self.max_retries} attempts failed for field " f"'{field_name or 'unknown'}': {e}"
                    )
                    return None

                # Calculate delay with exponential backoff
                delay = min(self.initial_delay * (self.backoff_factor**attempt), self.max_delay)
                logger.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)

        return None

    def batch_call(
        self,
        prompts: dict[str, str | dict],
        system_prompt: str | None = None,
    ) -> dict[str, str | None]:
        """Make multiple LLM calls for different fields.

        Args:
            prompts: Dictionary mapping field names to prompts. Each prompt can be:
                     - A string (legacy format, uses system_prompt parameter)
                     - A dict with 'system' and 'user' keys (new DSPy format)
            system_prompt: Optional fallback system prompt for legacy string prompts

        Returns:
            Dictionary mapping field names to responses
        """
        results = {}
        total = len(prompts)

        logger.info(f"Starting batch inference for {total} fields")

        for idx, (field_name, prompt) in enumerate(prompts.items(), 1):
            logger.info(f"Processing field {idx}/{total}: {field_name}")

            # Handle both new dict format and legacy string format
            if isinstance(prompt, dict):
                # New format: prompt is a dict with 'system' and 'user' keys
                field_system_prompt = prompt.get("system")
                user_prompt = prompt.get("user", "")
            else:
                # Legacy format: prompt is a string
                field_system_prompt = system_prompt
                user_prompt = prompt

            response = self.call(
                prompt=user_prompt,
                system_prompt=field_system_prompt,
                field_name=field_name,
            )

            results[field_name] = response

            if response is None:
                logger.warning(f"Failed to get response for field: {field_name}")

        successful = sum(1 for v in results.values() if v is not None)
        logger.info(f"Batch inference complete. Successful: {successful}/{total}")

        return results
