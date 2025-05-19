import os
from typing import Dict, Any, Optional
import json
from pathlib import Path
from dotenv import load_dotenv


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to config file (defaults to env variable CONFIG_PATH)
        
    Returns:
        Configuration dictionary
    """
    # Load environment variables
    load_dotenv()
    
    # Default config
    default_config = {
        "qdrant": {
            "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
            "api_key": os.getenv("QDRANT_API_KEY", "")
        },
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": os.getenv("OPENAI_MODEL", "text-embedding-3-small")
        },
        "collections": {
            "companies": {
                "dense_vector_size": 1536,
                "mini_vector_size": 384
            }
        },
        "search": {
            "default_limit": 10,
            "default_embedder": "dense"
        }
    }
    
    # If config path is provided, load and merge with default
    if config_path:
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                # Merge with default config (simple recursive merge)
                merge_configs(default_config, custom_config)
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            print("Using default config")
    
    return default_config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """
    Merge override config into base config (modifies base).
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            merge_configs(base[key], value)
        else:
            base[key] = value


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config to
    """
    try:
        # Create directory if it doesn't exist
        config_dir = os.path.dirname(config_path)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    # Get the module directory and go up to find project root
    current_dir = Path(__file__).parent
    
    # Go up until we find the project root (contains fild directory)
    while current_dir.name != "fild" and current_dir.parent != current_dir:
        current_dir = current_dir.parent
    
    # If we found the 'fild' directory, return its parent
    if current_dir.name == "fild":
        return current_dir.parent
    
    # Fallback: return the current working directory
    return Path.cwd() 