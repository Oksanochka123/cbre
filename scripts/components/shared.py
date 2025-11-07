"""Shared code for field extraction scripts."""


class DocumentProcessor:
    """A dummy class for processing documents."""

    def __init__(self, name: str, version: str = "1.0.0"):
        """Initialize a DocumentProcessor.

        Args:
            name: The name of the processor
            version: The version of the processor
        """
        self.name = name
        self.version = version

    def process(self, document: str) -> str:
        """Process a document.

        Args:
            document: The document to process

        Returns:
            The processed document
        """
        return f"Processed by {self.name} (v{self.version}): {document}"

    def __repr__(self) -> str:
        """Return string representation."""
        return f"DocumentProcessor(name={self.name!r}, version={self.version!r})"
