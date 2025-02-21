# Import necessary libraries
from sentence_transformers import SentenceTransformer
import torch
import pickle
import os
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import gc

# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Calculate embeddings for questions')
    parser.add_argument('--data_path', 
                        type=str,
                        default="/ephemeral/dobby_arena_logs.csv",
                        help='Path to the input CSV file')
    parser.add_argument('--output_dir',
                        type=str,
                        default="/ephemeral/query_embeddings_en_sentence_transformer",
                        help='Directory to save embeddings')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size for processing questions')
    return parser.parse_args()

# Get arguments
args = parse_args()
data_path = args.data_path
output_dir = args.output_dir
batch_size = args.batch_size

data = pd.read_csv(data_path)

# Load the model
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create output directory for embeddings
os.makedirs(output_dir, exist_ok=True)

# Find the highest existing index
existing_files = os.listdir(output_dir)
start_idx = 0
if existing_files:
    indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.startswith('embedding_')]
    if indices:
        start_idx = max(indices) + 1
        print(f"Starting from index {start_idx}")

# Function to get embeddings for a batch of texts
def get_batch_embeddings(texts):
    # Get embeddings
    with torch.no_grad():
        embeddings = model.encode(texts)
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    return embeddings

def save_embedding(args):
    idx, embedding, output_dir = args
    output_file = os.path.join(output_dir, f"embedding_{idx}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(embedding, f)

# Create a thread pool for file writing
max_workers = 4  # Adjust this based on your system
executor = ThreadPoolExecutor(max_workers=max_workers)

# Process in batches
for i in range(start_idx, len(data), batch_size):
    # Get batch of questions
    batch_questions = data['question'].iloc[i:i+batch_size].tolist()
    
    # Get embeddings for batch
    batch_embeddings = get_batch_embeddings(batch_questions)
    
    # Submit file writing tasks to thread pool
    futures = []
    for j, embedding in enumerate(batch_embeddings):
        idx = i + j
        future = executor.submit(save_embedding, (idx, embedding, output_dir))
        futures.append(future)
    
    # Clear batch from memory
    del batch_embeddings
    gc.collect()
    
    # Print progress
    print(f"Processed questions {i} to {min(i+batch_size, len(data))}")

# Clean up the thread pool
executor.shutdown(wait=True)