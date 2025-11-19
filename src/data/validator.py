"""Validate conversation data format."""

from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate conversation data meets requirements."""

    REQUIRED_FIELDS = ['context', 'student_message', 'tutor_response']
    OPTIONAL_FIELDS = ['conversation_id', 'turn_index', 'metadata']

    @staticmethod
    def validate_item(item: Dict, index: int) -> Tuple[bool, List[str]]:
        """Validate a single conversation item.

        Args:
            item: Conversation dictionary
            index: Index in dataset (for error messages)

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check required fields exist
        for field in DataValidator.REQUIRED_FIELDS:
            if field not in item:
                errors.append(f"Missing required field '{field}'")

        # Check types
        for field in DataValidator.REQUIRED_FIELDS:
            if field in item and not isinstance(item[field], str):
                errors.append(
                    f"Field '{field}' must be string, got {type(item[field])}"
                )

        # Check non-empty
        for field in DataValidator.REQUIRED_FIELDS:
            if field in item and not item[field].strip():
                errors.append(f"Field '{field}' cannot be empty")

        # Validate optional fields if present
        if 'turn_index' in item and not isinstance(item['turn_index'], int):
            errors.append(
                f"Field 'turn_index' must be int, got {type(item['turn_index'])}"
            )

        if 'metadata' in item and not isinstance(item['metadata'], dict):
            errors.append(
                f"Field 'metadata' must be dict, got {type(item['metadata'])}"
            )

        is_valid = len(errors) == 0
        return is_valid, errors

    @classmethod
    def validate_dataset(cls, data: List[Dict]) -> Tuple[bool, Dict]:
        """Validate entire dataset.

        Args:
            data: List of conversation dictionaries

        Returns:
            Tuple of (is_valid, validation_report)
        """
        report = {
            'total_items': len(data),
            'valid_items': 0,
            'invalid_items': 0,
            'errors': [],
            'warnings': []
        }

        if not isinstance(data, list):
            report['errors'].append("Data must be a list")
            return False, report

        if len(data) == 0:
            report['warnings'].append("Dataset is empty")
            return False, report

        # Validate each item
        for i, item in enumerate(data):
            is_valid, errors = cls.validate_item(item, i)

            if is_valid:
                report['valid_items'] += 1
            else:
                report['invalid_items'] += 1
                report['errors'].append({
                    'index': i,
                    'errors': errors,
                    'item_preview': str(item)[:100]
                })

        # Add warnings
        if report['valid_items'] < 50:
            report['warnings'].append(
                f"Dataset has only {report['valid_items']} valid items. "
                f"Recommended minimum: 50 for testing, 200+ for production."
            )

        is_valid = report['invalid_items'] == 0
        return is_valid, report


def validate_data_format(data: List[Dict]) -> bool:
    """Quick validation function.

    Args:
        data: List of conversation dictionaries

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    is_valid, report = DataValidator.validate_dataset(data)

    if not is_valid:
        error_msg = f"Data validation failed:\n"
        error_msg += f"  Valid items: {report['valid_items']}\n"
        error_msg += f"  Invalid items: {report['invalid_items']}\n"

        if report['errors']:
            error_msg += "\nFirst few errors:\n"
            for error in report['errors'][:3]:
                error_msg += f"  Item {error['index']}: {error['errors']}\n"

        if report['warnings']:
            error_msg += "\nWarnings:\n"
            for warning in report['warnings']:
                error_msg += f"  - {warning}\n"

        raise ValueError(error_msg)

    logger.info(f"✓ Data validation passed: {report['valid_items']} valid items")
    return True
