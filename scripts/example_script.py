"""Example script that uses shared code."""

from scripts.components.shared import DocumentProcessor


def main():
    """Main function to demonstrate usage of shared code."""
    processor = DocumentProcessor(name="example_processor", version="1.0.0")
    result = processor.process("Sample document")
    print(result)
    print(f"Processor: {processor}")


if __name__ == "__main__":
    main()
