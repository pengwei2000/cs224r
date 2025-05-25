# read json file and print the content
import json
def read_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            print(len(data))
            print(data[0])
            # print(json.dumps(data, indent=4))  # Pretty print the JSON content
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
read_json('ref_outputs.json')  # Replace 'data.json' with your actual file path