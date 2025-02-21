import shutil
from pathlib import Path
import concurrent.futures
import time

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
    source_directory = "/ephemeral/query_embeddings_en_sentence_transformer/"
    destination_directory = "/mnt/dobby-resources/arena_logs/query_embeddings_en_sentence_transformer/"
    fast_copy_files(source_directory, destination_directory, max_workers=4)