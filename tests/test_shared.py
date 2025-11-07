from scripts.components.shared import DocumentProcessor


class TestDocumentProcessor:
    def test_init_default_version(self):
        """Test initialization with default version."""
        processor = DocumentProcessor(name="test_processor")
        assert processor.name == "test_processor"
        assert processor.version == "1.0.0"

    def test_init_custom_version(self):
        """Test initialization with custom version."""
        processor = DocumentProcessor(name="test_processor", version="2.0.0")
        assert processor.name == "test_processor"
        assert processor.version == "2.0.0"

    def test_process(self):
        """Test document processing."""
        processor = DocumentProcessor(name="test_processor")
        result = processor.process("test document")
        assert "Processed by test_processor" in result
        assert "test document" in result

    def test_repr(self):
        """Test string representation."""
        processor = DocumentProcessor(name="test_processor", version="1.5.0")
        repr_str = repr(processor)
        assert "DocumentProcessor" in repr_str
        assert "test_processor" in repr_str
        assert "1.5.0" in repr_str
