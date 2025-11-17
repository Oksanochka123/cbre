"""Data loader for lease documents and ground truth."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class LeaseDataLoader:
    """Loader for lease document data."""

    def __init__(
        self,
        data_dir: Path,
        vlm_suffix: str = "_vlm.txt",
        meta_suffix: str = "_meta.json",
        file_separator: str = "\n\n" + "=" * 80 + "\n\n",
        include_filename: bool = True,
    ):
        """Initialize the data loader.

        Args:
            data_dir: Root directory containing lease subfolders
            vlm_suffix: Suffix for VLM text files
            meta_suffix: Suffix for metadata files (to exclude)
            file_separator: Separator to use between multiple documents
            include_filename: Whether to include filename before each document
        """
        self.data_dir = Path(data_dir)
        self.vlm_suffix = vlm_suffix
        self.meta_suffix = meta_suffix
        self.file_separator = file_separator
        self.include_filename = include_filename

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        logger.info(f"Initialized data loader for directory: {self.data_dir}")

    def list_lease_folders(self) -> list[Path]:
        """List all lease subfolders in the data directory.

        Returns:
            List of paths to lease subfolders
        """
        folders = [f for f in self.data_dir.iterdir() if f.is_dir() and not f.name.startswith(".")]

        logger.info(f"Found {len(folders)} lease folders")
        return sorted(folders)

    def load_lease_text(self, lease_folder: Path) -> str | None:
        """Load and concatenate all VLM text files for a lease.

        Args:
            lease_folder: Path to the lease subfolder

        Returns:
            Concatenated text from all VLM files, or None if no files found
        """
        vlm_files = sorted(lease_folder.glob(f"*{self.vlm_suffix}"))

        if not vlm_files:
            logger.warning(f"No VLM files found in {lease_folder.name}")
            return None

        logger.debug(f"Found {len(vlm_files)} VLM files in {lease_folder.name}")

        documents = []
        for vlm_file in vlm_files:
            try:
                with open(vlm_file, encoding="utf-8") as f:
                    content = f.read().strip()

                if self.include_filename:
                    doc_text = f"Filename: {vlm_file.name}\n\n{content}"
                else:
                    doc_text = content

                documents.append(doc_text)

                logger.debug(f"Loaded {len(content)} chars from {vlm_file.name}")

            except Exception as e:
                logger.error(f"Error reading {vlm_file}: {e}")
                continue

        if not documents:
            logger.warning(f"Could not load any documents from {lease_folder.name}")
            return None

        # Concatenate all documents with separator
        full_text = self.file_separator.join(documents)

        logger.info(
            f"Loaded {len(vlm_files)} documents from {lease_folder.name}, " f"total length: {len(full_text)} chars"
        )

        return full_text

    def find_ground_truth_json(self, lease_folder: Path) -> Path | None:
        """Find the ground truth JSON file in a lease folder.

        Excludes files ending with _meta.json.
        If multiple JSON files exist, selects the one with the most recent date in filename.

        Args:
            lease_folder: Path to the lease subfolder

        Returns:
            Path to ground truth JSON file, or None if not found
        """
        # Find all JSON files, excluding metadata files
        json_files = [f for f in lease_folder.glob("*.json") if not f.name.endswith(self.meta_suffix)]

        if not json_files:
            logger.debug(f"No ground truth JSON found in {lease_folder.name}")
            return None

        if len(json_files) == 1:
            logger.debug(f"Found ground truth JSON: {json_files[0].name}")
            return json_files[0]

        # Multiple JSON files - select the one with most recent date
        logger.info(f"Found {len(json_files)} JSON files in {lease_folder.name}, " "selecting most recent")

        json_with_dates = []
        for json_file in json_files:
            date = self._extract_date_from_filename(json_file.name)
            json_with_dates.append((json_file, date))

        # Sort by date (None dates go first, then by date descending)
        json_with_dates.sort(key=lambda x: (x[1] is None, x[1] if x[1] else datetime.min), reverse=True)

        selected = json_with_dates[0][0]
        logger.info(f"Selected ground truth JSON: {selected.name}")
        return selected

    def _extract_date_from_filename(self, filename: str) -> datetime | None:
        """Extract date from filename using common patterns.

        Tries multiple date formats commonly found in filenames.

        Args:
            filename: The filename to parse

        Returns:
            Parsed datetime object, or None if no date found
        """
        # Common date patterns in filenames
        patterns = [
            r"(\d{4})[-_](\d{2})[-_](\d{2})",  # YYYY-MM-DD or YYYY_MM_DD
            r"(\d{2})[-_](\d{2})[-_](\d{4})",  # MM-DD-YYYY or MM_DD_YYYY
            r"(\d{4})(\d{2})(\d{2})",  # YYYYMMDD
            r"(\d{2})(\d{2})(\d{4})",  # MMDDYYYY
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                groups = match.groups()
                try:
                    # Try YYYY-MM-DD format
                    if len(groups[0]) == 4:
                        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                    # Try MM-DD-YYYY format
                    else:
                        month, day, year = int(groups[0]), int(groups[1]), int(groups[2])

                    return datetime(year, month, day)
                except ValueError:
                    continue

        return None

    def load_ground_truth_json(self, json_path: Path) -> dict | None:
        """Load ground truth JSON data.

        Args:
            json_path: Path to the JSON file

        Returns:
            Parsed JSON data, or None if loading failed
        """
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            logger.debug(f"Loaded ground truth from {json_path.name}")
            return data

        except Exception as e:
            logger.error(f"Error loading ground truth from {json_path}: {e}")
            return None

    def load_lease_data(self, lease_folder: Path) -> tuple[str, dict | None] | None:
        """Load both lease text and ground truth for a lease folder.

        Args:
            lease_folder: Path to the lease subfolder

        Returns:
            Tuple of (lease_text, ground_truth_data), or None if lease text not found
        """
        # Load lease text
        lease_text = self.load_lease_text(lease_folder)
        if not lease_text:
            return None

        # Try to load ground truth
        gt_json_path = self.find_ground_truth_json(lease_folder)
        ground_truth = None
        if gt_json_path:
            ground_truth = self.load_ground_truth_json(gt_json_path)

        return lease_text, ground_truth
