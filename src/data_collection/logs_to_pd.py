import json
import os
import fasttext
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import pandas as pd
import argparse

# Add argument parser
parser = argparse.ArgumentParser(description='Process Dobby Arena logs into a CSV file')
parser.add_argument('--data_dir', 
                    type=str,
                    default="/ephemeral/dobby_arena_english_logs",
                    help='Directory containing the JSON log files')
parser.add_argument('--csv_out_path', 
                    type=str,
                    default="/ephemeral/dobby_arena_logs.csv",
                    help='Path where the output CSV file will be saved')

args = parser.parse_args()

# Use the arguments instead of hardcoded values
data_dir = args.data_dir
# Initialize lists to store the data
chat_ids = []
entry_ids = []
questions = []  # New list for questions
dobby_unhinged_texts = []
dobby_leashed_texts = []
votes_unhinged = []
votes_leashed = []
json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

with open(os.path.join(data_dir, json_files[0]), 'r') as f:
    # Read line by line and parse each JSON object
    data = [json.loads(line.strip()) for line in f if line.strip()]

# Counter for unique entry IDs
entry_id_counter = 0

# Loop through all JSON files with tqdm
for json_file in tqdm(json_files, desc="Processing files"):
    with open(os.path.join(data_dir, json_file), 'r') as f:
        # Read all lines first to show progress for interactions
        lines = [line.strip() for line in f if line.strip()]
        
        for line in tqdm(lines, desc=f"Processing interactions in {json_file}", leave=False):
            data = json.loads(line)
            chat_id = data['messages']['chatId']
            interactions = data['messages']['interactions']
            
            # Process each interaction
            for interaction_id, interaction in interactions.items():
                entry_id_counter += 1
                
                # Get the question and answers
                question = interaction.get('question', '')  # Get the question
                answers = interaction.get('answers', {})
                dobby_unhinged = answers.get('dobby-unhinged', '')
                dobby_leashed = answers.get('dobby', '')
                
                # Store the data
                chat_ids.append(chat_id)
                entry_ids.append(entry_id_counter)
                questions.append(question)  # Add question to the list
                dobby_unhinged_texts.append(dobby_unhinged)
                dobby_leashed_texts.append(dobby_leashed)
                
                # Check for votes
                votes = data['messages'].get('vote', [])
                vote_unhinged = None
                vote_leashed = None
                
                for vote in votes:
                    if vote['messageIndex'] == interaction_id:
                        vote_unhinged = vote.get('dobbyUnhinged')
                        vote_leashed = vote.get('dobby')
                
                votes_unhinged.append(vote_unhinged)
                votes_leashed.append(vote_leashed)

# Create the DataFrame
df = pd.DataFrame({
    'chat_id': chat_ids,
    'entry_id': entry_ids,
    'question': questions,  # Add questions to DataFrame
    'dobby_unhinged': dobby_unhinged_texts,
    'dobby_leashed': dobby_leashed_texts,
    'vote_unhinged': votes_unhinged,
    'vote_leashed': votes_leashed
})

# Save the DataFrame to a CSV file
df.to_csv(args.csv_out_path, index=False)

print(f"DataFrame saved to '{args.csv_out_path}'")
