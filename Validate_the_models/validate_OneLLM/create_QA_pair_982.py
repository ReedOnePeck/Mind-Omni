import numpy as np
import json
import os
from tqdm import tqdm


def process_and_extract_q_and_a():
    """
    Loads a list of filenames, finds corresponding JSON files, extracts question-answer
    pairs, and saves them to separate files. Reports any missing files.
    """
    # --- 1. Define all necessary file paths ---
    # Input file containing the 982 target filenames
    matched_names_path = '/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/validate_models/matched_names_by_CLIP.npy'

    # Directory where the source JSON files are located
    source_json_dir = '/data/home/luyizhuo/Datastation_lyz/Datasets/NSD_complete/Visual_instruct_tuning_data/recaptioned_data/complex_reasoning'

    # Output file paths
    output_questions_path = '/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/validate_models/982_reason_Q.json'
    output_answers_path = '/data/home/luyizhuo/Datastation_lyz/Models/UniBrain/validate_models/982_reason_A.json'

    # --- 2. Load the target filenames ---
    print(f"Loading target filenames from: {matched_names_path}")
    try:
        target_filenames = np.load(matched_names_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"FATAL ERROR: Input file not found at {matched_names_path}. Please check the path.")
        return

    print(f"Successfully loaded {len(target_filenames)} filenames.")

    # --- 3. Process each filename ---
    all_questions = []
    all_answers = []
    unmatched_indices = []

    print(f"Searching for JSON files in: {source_json_dir}")
    # Use tqdm for a nice progress bar
    for i, png_filename in enumerate(tqdm(target_filenames, desc="Processing files")):
        # Derive the JSON filename from the PNG filename (e.g., '01234.png' -> '01234.json')
        base_name = os.path.splitext(png_filename)[0]
        json_filename = f"{base_name}.json"
        json_filepath = os.path.join(source_json_dir, json_filename)

        if os.path.exists(json_filepath):
            try:
                with open(json_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # The data is a list containing one dictionary
                    if data and isinstance(data, list) and len(data) > 0:
                        question = data[0].get("Question", "Question key not found")
                        answer = data[0].get("Answer", "Answer key not found")
                        all_questions.append(question)
                        all_answers.append(answer)
                    else:
                        # Handle cases where the JSON file is empty or has an unexpected format
                        all_questions.append("JSON file is empty or invalid format")
                        all_answers.append("JSON file is empty or invalid format")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not process file {json_filename}. Error: {e}")
                all_questions.append(f"Error processing JSON: {e}")
                all_answers.append(f"Error processing JSON: {e}")
        else:
            # If the JSON file does not exist, record the index and add placeholders
            unmatched_indices.append(i)
            all_questions.append("Corresponding JSON file not found.")
            all_answers.append("Corresponding JSON file not found.")

    # --- 4. Save the results in the specified JSON Lines format ---
    def save_as_json_lines(data_list, output_path):
        """Helper function to save a list of strings in the specified format."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for text in data_list:
                    # Create the dictionary structure
                    line_dict = {"prompt": text}
                    # Convert dict to a JSON string and write it, followed by a newline
                    f.write(json.dumps(line_dict) + '\n')
            print(f"Successfully saved {len(data_list)} items to: {output_path}")
        except Exception as e:
            print(f"ERROR: Could not save file to {output_path}. Reason: {e}")

    # Save the questions
    save_as_json_lines(all_questions, output_questions_path)

    # Save the answers
    save_as_json_lines(all_answers, output_answers_path)

    # --- 5. Report any unmatched files ---
    print("\n--- Processing Complete ---")
    if not unmatched_indices:
        print("Success! All 982 filenames were successfully matched to their corresponding JSON files.")
    else:
        print(f"Warning: Could not find matching JSON files for {len(unmatched_indices)} item(s).")
        print("Indices of missing files are listed below:")
        for index in unmatched_indices:
            print(f"  - Index: {index}, Filename: {target_filenames[index]}")
    print(unmatched_indices)


if __name__ == '__main__':
    process_and_extract_q_and_a()