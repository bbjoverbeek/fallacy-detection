import json
import os
import re

fallacy_codes = ['AA', 'BB', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'JJ', 'KK', 'LL', 'MM', 'NN', 'OO', 'PP', 'QQ', 'RR', 'SS', 'TT', 'UU', 'VV', 'WW', 'XX', 'YY', 'ZZ', 'AB', 'AC', 'AD']


def find_double_answers(model_outputs):
    # Initialize a set to store unique valid codes found
    unique_codes_found = set()

    # Ensure the output is a string and convert it to upper case for uniformity
    output = model_outputs[0].upper() if isinstance(model_outputs, list) else model_outputs.upper()

    # Extract potential codes using regular expression to split by non-alphanumeric characters
    potential_codes = re.split(r'\W+', output)

    # Store each valid code into a set
    for code in potential_codes:
        if code in fallacy_codes:
            unique_codes_found.add(code)

    # Check if at least two different codes are found
    return len(unique_codes_found) >= 2


def load_all_json_files(folder_path):
    json_files_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                json_files_data[file_name] = json.load(file)
    return json_files_data

def check_double_answers(json_files_data):
    for data in json_files_data.values():
        for key, value in data.items():
            try:
                # convert key to int
                key = int(key)
            except ValueError:
                continue
            output = value['model_outputs']

            if len(output) == 1:
                answers = find_double_answers(output)
            elif len(output) > 1:
                # skip self consistency for now
                pass
            if answers:
                print(f"Double answers found in {key}")
def main():
    folder_path = 'results'
    json_files_data = load_all_json_files(folder_path)
    check_double_answers(json_files_data)


if __name__ == '__main__':
    main()