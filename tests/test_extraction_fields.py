#!/usr/bin/env python3
"""
Validate that all fields in YAML config can be extracted from annotation JSON files.

This test ensures:
1. All fields have valid json_path defined
2. JSON paths match the annotation structure
3. No typos in section names or field paths
4. Table structures are correct

Usage:
    python test_field_extraction.py \
        --config configs/fields_config.yaml \
        --annotation data/raw/annotations/sample.json

    # Or test all annotations
    python test_field_extraction.py \
        --config configs/fields_config.yaml \
        --annotations_dir data/raw/annotations
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


class FieldValidator:
    """Validator for field extraction from JSON annotations."""
    
    def __init__(self, config_path: Path):
        """Initialize validator with config.
        
        Args:
            config_path: Path to YAML config
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.fields = self.config.get('fields', {})
        self.results = {
            'passed': [],
            'failed': [],
            'missing_path': [],
            'warnings': []
        }
    
    def extract_value(self, annotation: Dict[str, Any], json_path: str) -> Tuple[Any, str]:
        """Extract value using JSON path.
        
        Args:
            annotation: Annotation JSON
            json_path: JSON path string
            
        Returns:
            Tuple of (value, error_message)
        """
        try:
            parts = json_path.split('::')
            
            if len(parts) < 3:
                return None, f"Invalid path format (expected 3+ parts, got {len(parts)})"
            
            path_type = parts[0]
            section = parts[1]
            
            # Check section exists
            if section not in annotation:
                return None, f"Section '{section}' not found in annotation"
            
            section_data = annotation[section]
            
            if path_type == "STATIC":
                field_path = parts[2]
                
                # Check static_fields exists
                if 'static_fields' not in section_data:
                    return None, f"'static_fields' not found in section '{section}'"
                
                static_fields = section_data['static_fields']
                
                # Check field exists
                if field_path not in static_fields:
                    # Try to find similar fields
                    similar = [f for f in static_fields.keys() if field_path.lower() in f.lower()]
                    if similar:
                        return None, f"Field '{field_path}' not found. Similar fields: {similar[:3]}"
                    return None, f"Field '{field_path}' not found in static_fields"
                
                return static_fields[field_path], None
                
            elif path_type == "TABLE":
                table_name = parts[2]
                
                # Check tables exists
                if 'tables' not in section_data:
                    return None, f"'tables' not found in section '{section}'"
                
                tables = section_data['tables']
                
                # Check table exists
                if table_name not in tables:
                    # Try to find similar tables
                    similar = [t for t in tables.keys() if table_name.lower() in t.lower()]
                    if similar:
                        return None, f"Table '{table_name}' not found. Similar tables: {similar[:3]}"
                    return None, f"Table '{table_name}' not found in tables"
                
                table_data = tables[table_name]
                
                # Validate it's a list
                if not isinstance(table_data, list):
                    return None, f"Table '{table_name}' is not a list (type: {type(table_data).__name__})"
                
                return table_data, None
            
            else:
                return None, f"Unknown path type: {path_type}"
                
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"
    
    def validate_field(self, field_name: str, config: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single field.
        
        Args:
            field_name: Name of field
            config: Field configuration
            annotation: Annotation JSON
            
        Returns:
            Validation result dictionary
        """
        result = {
            'field': field_name,
            'type': config.get('type'),
            'wave': config.get('wave'),
            'json_path': config.get('json_path'),
            'status': 'unknown',
            'message': '',
            'value_preview': None
        }
        
        # Check if json_path is defined
        json_path = config.get('json_path')
        if not json_path:
            result['status'] = 'missing_path'
            result['message'] = "No json_path defined in config"
            return result
        
        # Try to extract
        value, error = self.extract_value(annotation, json_path)
        
        if error:
            result['status'] = 'failed'
            result['message'] = error
            return result
        
        # Success
        result['status'] = 'passed'
        
        # Add value preview
        if value is None:
            result['message'] = "Extracted successfully (value is None)"
        elif isinstance(value, list):
            result['message'] = f"Extracted successfully ({len(value)} items)"
            result['value_preview'] = value[:2] if len(value) > 0 else []
        elif isinstance(value, str):
            preview = value[:50] + "..." if len(value) > 50 else value
            result['message'] = f"Extracted successfully"
            result['value_preview'] = preview
        else:
            result['message'] = f"Extracted successfully (type: {type(value).__name__})"
            result['value_preview'] = str(value)[:50]
        
        return result
    
    def validate_all_fields(self, annotation: Dict[str, Any]) -> None:
        """Validate all fields in config against annotation.
        
        Args:
            annotation: Annotation JSON
        """
        print(f"\n{'='*80}")
        print(f"Validating {len(self.fields)} fields")
        print(f"{'='*80}\n")
        
        for field_name, field_config in self.fields.items():
            result = self.validate_field(field_name, field_config, annotation)
            
            status = result['status']
            if status == 'passed':
                self.results['passed'].append(result)
            elif status == 'failed':
                self.results['failed'].append(result)
            elif status == 'missing_path':
                self.results['missing_path'].append(result)
    
    def print_results(self) -> bool:
        """Print validation results.
        
        Returns:
            True if all fields passed, False otherwise
        """
        print(f"\n{'='*80}")
        print("VALIDATION RESULTS")
        print(f"{'='*80}\n")
        
        total = len(self.fields)
        passed = len(self.results['passed'])
        failed = len(self.results['failed'])
        missing = len(self.results['missing_path'])
        
        print(f"Total fields:     {total}")
        print(f"✓ Passed:         {passed} ({100*passed/total:.1f}%)")
        print(f"✗ Failed:         {failed} ({100*failed/total:.1f}%)")
        print(f"⚠ Missing path:   {missing} ({100*missing/total:.1f}%)")
        
        # Show failed fields
        if self.results['failed']:
            print(f"\n{'='*80}")
            print(f"FAILED FIELDS ({len(self.results['failed'])})")
            print(f"{'='*80}\n")
            
            for result in self.results['failed']:
                print(f"✗ {result['field']} (Wave {result['wave']}, {result['type']})")
                print(f"  Path: {result['json_path']}")
                print(f"  Error: {result['message']}")
                print()
        
        # Show missing path fields
        if self.results['missing_path']:
            print(f"\n{'='*80}")
            print(f"FIELDS WITHOUT JSON_PATH ({len(self.results['missing_path'])})")
            print(f"{'='*80}\n")
            
            for result in self.results['missing_path']:
                print(f"⚠ {result['field']} (Wave {result['wave']}, {result['type']})")
        
        # Show successful extractions (sample)
        if self.results['passed']:
            print(f"\n{'='*80}")
            print(f"SAMPLE SUCCESSFUL EXTRACTIONS (showing 10/{len(self.results['passed'])})")
            print(f"{'='*80}\n")
            
            for result in self.results['passed'][:10]:
                print(f"✓ {result['field']}: {result['message']}")
                if result['value_preview']:
                    print(f"  Value: {result['value_preview']}")
                print()
        
        # Summary
        print(f"\n{'='*80}")
        all_good = failed == 0 and missing == 0
        if all_good:
            print("🎉 ALL FIELDS VALIDATED SUCCESSFULLY!")
        else:
            print("⚠️  VALIDATION ISSUES FOUND - See details above")
        print(f"{'='*80}\n")
        
        return all_good
    
    def get_annotation_structure(self, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """Get structure summary of annotation.
        
        Args:
            annotation: Annotation JSON
            
        Returns:
            Structure summary
        """
        structure = {}
        
        for section_name, section_data in annotation.items():
            section_info = {}
            
            if 'static_fields' in section_data:
                section_info['static_fields_count'] = len(section_data['static_fields'])
                section_info['static_fields_sample'] = list(section_data['static_fields'].keys())[:5]
            
            if 'tables' in section_data:
                tables_info = {}
                for table_name, table_data in section_data['tables'].items():
                    if isinstance(table_data, list) and table_data:
                        tables_info[table_name] = {
                            'row_count': len(table_data),
                            'columns': list(table_data[0].keys())
                        }
                section_info['tables'] = tables_info
            
            structure[section_name] = section_info
        
        return structure


def main():
    parser = argparse.ArgumentParser(description="Validate field extraction from JSON annotations")
    parser.add_argument('--config', required=True, help='Path to YAML config')
    
    # Either single annotation or directory
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--annotation', help='Path to single annotation JSON')
    group.add_argument('--annotations_dir', help='Directory with annotation files')
    
    parser.add_argument('--show_structure', action='store_true', 
                       help='Show annotation structure before validation')
    
    args = parser.parse_args()
    
    # Load validator
    config_path = Path(args.config)
    validator = FieldValidator(config_path)
    
    # Get annotation file(s)
    if args.annotation:
        annotation_files = [Path(args.annotation)]
    else:
        annotations_dir = Path(args.annotations_dir)
        annotation_files = sorted(annotations_dir.glob('*.json'))
        print(f"Found {len(annotation_files)} annotation files")
        
        if len(annotation_files) > 1:
            print(f"Using first file for validation: {annotation_files[0].name}")
        
        if not annotation_files:
            print("Error: No annotation files found!")
            return 1
    
    # Load annotation
    annotation_path = annotation_files[0]
    print(f"Loading annotation: {annotation_path}")
    
    with open(annotation_path) as f:
        annotation = json.load(f)
    
    # Show structure if requested
    if args.show_structure:
        print(f"\n{'='*80}")
        print("ANNOTATION STRUCTURE")
        print(f"{'='*80}\n")
        
        structure = validator.get_annotation_structure(annotation)
        print(json.dumps(structure, indent=2))
    
    # Validate all fields
    validator.validate_all_fields(annotation)
    
    # Print results
    success = validator.print_results()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())