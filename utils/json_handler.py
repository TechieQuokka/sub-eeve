"""
JSON file handling utilities
"""
import json
from pathlib import Path
from typing import Dict, Any
from config import FIELD_NAME_MAP


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON subtitle file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary containing subtitle data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'segments' not in data:
        raise ValueError("JSON file must contain 'segments' key")

    return data


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to JSON file.

    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_field_name(target_lang: str) -> str:
    """
    Get field name for target language.

    Args:
        target_lang: Target language code (ko, en, ja, zh)

    Returns:
        Field name for the target language (e.g., 'text_ko')
    """
    return FIELD_NAME_MAP.get(target_lang, f'text_{target_lang}')
