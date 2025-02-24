import shutil
from pathlib import Path
import concurrent.futures
import time
import argparse

def copy_file(src_file: Path, dst_dir: Path):
    """Copy a single file if it doesn't exist in destination."""
    dst_file = dst_dir / src_file.name
    if not dst_file.exists():
        shutil.copy2(src_file, dst_file)
        return f"Copied: {src_file.name}"
    return f"Skipped (exists): {src_file.name}"

def fast_copy_files(src_dir: str, dst_dir: str, max_workers: int = 4):
    """
    Quickly copy files from source to destination directory using parallel processing.
    Only copies files that don't exist in the destination.
    """
    # Convert to Path objects
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # Create destination directory if it doesn't exist
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of files in source directory
    src_files = list(src_path.glob('*'))
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel copying
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_file, file, dst_path) for file in src_files]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
    
    end_time = time.time()
    print(f"\nCopy completed in {end_time - start_time:.2f} seconds")
    print(f"Processed {len(src_files)} files")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fast parallel file copy utility')
    parser.add_argument('source', help='Source directory path')
    parser.add_argument('destination', help='Destination directory path')
    parser.add_argument('--workers', type=int, default=4, 
                       help='Number of worker threads (default: 4)')
    
    args = parser.parse_args()
    
    fast_copy_files(args.source, args.destination, max_workers=args.workers)