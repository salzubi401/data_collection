import json
import os
import fasttext
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import argparse

# Add argument parser before data_dir definition
parser = argparse.ArgumentParser(description='Filter English logs from conversation data')
parser.add_argument('--data-dir', 
                   default="/mnt/arena_bucket/dobby_conversation_log/arena/user_info_less_messages/06022025-2300-ist/",
                   help='Directory containing the log files to process')
parser.add_argument('--output-dir',
                   default="/ephemeral/dobby_arena_english_logs",
                   help='Directory where filtered English logs will be saved')
parser.add_argument('--n-files', 
                   type=int,
                   default=-1,
                   help='Number of files to process. Use -1 to process all files')
parser.add_argument('--confidence-threshold',
                   type=float,
                   default=0.5,
                   help='Confidence threshold for English language detection')
args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir
N_FILES = args.n_files
ENGLISH_CONFIDENCE_THRESHOLD = args.confidence_threshold

model_path = hf_hub_download(
    repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)
json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

# Process all files if N_FILES is -1, otherwise process the first N_FILES
files_to_process = json_files if N_FILES == -1 else json_files[:N_FILES]

total_entries = 0
total_english_entries = 0

# Process files
for json_file in tqdm(files_to_process, desc="Processing files"):
    input_path = os.path.join(data_dir, json_file)
    output_path = os.path.join(output_dir, json_file)
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(input_path, 'r') as f:
            # Read line by line and parse each JSON object
            data = [json.loads(line.strip()) for line in f if line.strip()]
        
        english_logs = []
        # Process each log entry
        for entry in data:
            total_entries += 1
            if 'messages' in entry and 'interactions' in entry['messages']:
                # Calculate average confidence score for all questions
                total_score = 0
                question_count = 0
                for _, interaction in entry['messages']['interactions'].items():
                    # Clean the question text by replacing newlines with spaces
                    question = interaction['question'].replace('\n', ' ').strip()
                    prediction = model.predict(question)
                    if prediction[0][0] == '__label__eng_Latn':
                        total_score += prediction[1][0]
                    question_count += 1
                
                # Add entry if average confidence score is above threshold
                if question_count > 0 and (total_score / question_count) >= ENGLISH_CONFIDENCE_THRESHOLD:
                    english_logs.append(entry)
                    total_english_entries += 1
        
        # Save filtered logs if we found any English entries
        if english_logs:
            with open(output_path, 'w') as f:
                for entry in english_logs:
                    f.write(json.dumps(entry) + '\n')
            
    except Exception as e:
        print(f"Error processing file {json_file}: {str(e)}")

english_percentage = (total_english_entries / total_entries * 100) if total_entries > 0 else 0
print(f"Processed {N_FILES} files. Check the '{output_dir}' directory for results.")
print(f"Found {total_english_entries} English entries out of {total_entries} total entries ({english_percentage:.2f}%)")