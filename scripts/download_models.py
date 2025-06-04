#!/usr/bin/env python3
"""
Script to pre-download SentenceTransformer models during Docker build.
This prevents runtime downloads and disk space issues.
"""

import os
import sys
import shutil
import tempfile
from sentence_transformers import SentenceTransformer

def get_disk_space(path):
    """Get available disk space in MB."""
    try:
        total, used, free = shutil.disk_usage(path)
        return free // (1024 * 1024)  # Convert to MB
    except Exception:
        return 0

def download_model(model_name, min_space_mb=1000):
    """Download a single model with error handling and space check."""
    try:
        cache_dir = os.getenv('SENTENCE_TRANSFORMERS_HOME', '/app/model_cache')
        free_space = get_disk_space(cache_dir)
        
        print(f"Downloading model: {model_name}")
        print(f"Available space: {free_space} MB")
        
        if free_space < min_space_mb:
            print(f"⚠️  Skipping {model_name}: insufficient disk space ({free_space} MB < {min_space_mb} MB)")
            return False
            
        model = SentenceTransformer(model_name)
        print(f"✅ Successfully downloaded: {model_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

def main():
    """Download only essential models."""
    print("Pre-downloading essential SentenceTransformer models...")
    
    # Only download the most critical model used by SentenceTransformerShort
    essential_models = [
        "sentence-transformers/all-MiniLM-L6-v2",  # Primary model used in the app
    ]
    
    # Optional models - only download if we have enough space
    optional_models = [
        "sentence-transformers/all-mpnet-base-v2",  # Used in experiments
    ]
    
    print(f"Essential models to download: {len(essential_models)}")
    print(f"Optional models: {len(optional_models)}")
    
    success_count = 0
    failed_models = []
    
    # Download essential models first
    for model_name in essential_models:
        if download_model(model_name, min_space_mb=500):  # Lower threshold for essential
            success_count += 1
        else:
            failed_models.append(model_name)
    
    # Download optional models if we have space
    for model_name in optional_models:
        if download_model(model_name, min_space_mb=1500):  # Higher threshold for optional
            success_count += 1
        else:
            print(f"⚠️  Skipped optional model: {model_name}")
    
    print(f"\nDownload summary:")
    print(f"✅ Successfully downloaded: {success_count} models")
    
    if failed_models:
        print(f"❌ Failed essential downloads: {failed_models}")
        # Don't fail the build for optional models, only essential ones
        if any(model in essential_models for model in failed_models):
            print("💥 Build failed: Essential models could not be downloaded")
            sys.exit(1)
    
    print("🎉 Essential models downloaded successfully!")
    
    # Show final cache size
    cache_dir = os.getenv('SENTENCE_TRANSFORMERS_HOME', '/app/model_cache')
    if os.path.exists(cache_dir):
        try:
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(cache_dir)
                for filename in filenames
            )
            print(f"📦 Model cache size: {total_size // (1024 * 1024)} MB")
        except Exception:
            pass

if __name__ == "__main__":
    main() 