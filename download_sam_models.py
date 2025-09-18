#!/usr/bin/env python3
"""
Download SAM model checkpoints from Meta's official repository
Downloads all three model sizes: ViT-H, ViT-L, and ViT-B
"""

import os
import requests
from pathlib import Path
import sys

# Official SAM model checkpoint URLs
SAM_MODELS = {
    "sam_vit_h_4b8939.pth": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "size": "~2.6GB",
        "type": "vit_h",
        "description": "ViT-H SAM model (Largest, most accurate)"
    },
    "sam_vit_l_0b3195.pth": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
        "size": "~1.2GB",
        "type": "vit_l",
        "description": "ViT-L SAM model (Medium size, good balance)"
    },
    "sam_vit_b_01ec64.pth": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "size": "~375MB", 
        "type": "vit_b",
        "description": "ViT-B SAM model (Smallest, fastest)"
    }
}

def download_file(url, filepath, description):
    """Download a file with progress indication"""
    print(f"\n📥 Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   Saving to: {filepath}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Simple progress indicator
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Progress: {percent:.1f}% ({downloaded:,} / {total_size:,} bytes)", end='')
                    else:
                        print(f"\r   Downloaded: {downloaded:,} bytes", end='')
        
        print(f"\n✅ Successfully downloaded {description}")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {description}: {e}")
        return False

def main():
    print("🤖 SAM Model Checkpoint Downloader")
    print("=" * 50)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"📁 Models directory: {models_dir.absolute()}")
    
    # Check available space (rough estimate)
    total_size_gb = 2.6 + 1.2 + 0.375  # Approximate total size
    print(f"💾 Total download size: ~{total_size_gb:.1f}GB")
    
    # Ask user which models to download
    print("\n📋 Available SAM models:")
    for i, (filename, info) in enumerate(SAM_MODELS.items(), 1):
        print(f"   {i}. {filename} ({info['size']}) - {info['description']}")
    
    print("\n🔽 Download options:")
    print("   1. Download all models (recommended)")
    print("   2. Download ViT-B only (fastest, smallest)")
    print("   3. Download ViT-L only (balanced)")
    print("   4. Download ViT-H only (best quality)")
    print("   5. Custom selection")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
    except KeyboardInterrupt:
        print("\n\n❌ Download cancelled by user")
        return False
    
    # Determine which models to download
    models_to_download = []
    
    if choice == "1":
        models_to_download = list(SAM_MODELS.keys())
    elif choice == "2":
        models_to_download = ["sam_vit_b_01ec64.pth"]
    elif choice == "3":
        models_to_download = ["sam_vit_l_0b3195.pth"]
    elif choice == "4":
        models_to_download = ["sam_vit_h_4b8939.pth"]
    elif choice == "5":
        print("\nSelect models to download (y/n):")
        for filename, info in SAM_MODELS.items():
            response = input(f"   Download {filename} ({info['size']})? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                models_to_download.append(filename)
    else:
        print("❌ Invalid choice")
        return False
    
    if not models_to_download:
        print("❌ No models selected for download")
        return False
    
    # Download selected models
    successful_downloads = 0
    
    for filename in models_to_download:
        filepath = models_dir / filename
        
        # Skip if file already exists
        if filepath.exists():
            print(f"\n⏭️  {filename} already exists, skipping...")
            successful_downloads += 1
            continue
        
        info = SAM_MODELS[filename]
        success = download_file(info["url"], filepath, f"{filename} ({info['size']})")
        
        if success:
            successful_downloads += 1
        else:
            # Clean up partial download
            if filepath.exists():
                filepath.unlink()
    
    # Summary
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successful: {successful_downloads}/{len(models_to_download)}")
    print(f"   📁 Location: {models_dir.absolute()}")
    
    if successful_downloads > 0:
        print(f"\n🎉 Ready to use SAM models!")
        print(f"   Set SAM_MODELS_DIR=models in your environment")
        print(f"   Or select models in the Streamlit interface")
        
        # Show usage example
        print(f"\n💡 Usage in Python:")
        for filename in models_to_download:
            if (models_dir / filename).exists():
                model_type = SAM_MODELS[filename]["type"]
                print(f"   # {SAM_MODELS[filename]['description']}")
                print(f"   sam = sam_model_registry['{model_type}'](checkpoint='models/{filename}')")
                break
    
    return successful_downloads > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)