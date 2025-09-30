import json

# Convert JSON file to JSONL format

def json_to_jsonl(json_filepath, jsonl_filepath):
    with open(json_filepath, 'r') as infile, open(jsonl_filepath, 'w') as outfile:
        data = json.load(infile)
        if isinstance(data, list):  # Check if it's a JSON array
            for item in data:
                json.dump(item, outfile)
                outfile.write('\n')
        else: # Handle single JSON object if needed
            json.dump(data, outfile)

# json_to_jsonl('data/esa_scores_X_ALMA.json', 'data/esa_scores_X_ALMA.jsonl')