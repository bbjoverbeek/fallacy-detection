import json
import os

fallacy_codes = ['AA', 'BB', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'JJ', 'KK', 'LL', 'MM', 'NN', 'OO', 'PP', 'QQ', 'RR', 'SS', 'TT', 'UU', 'VV', 'WW', 'XX', 'YY', 'ZZ', 'AB', 'AC', 'AD']
def find_double_answers(model_outputs):
    # Find distinct fallacy codes in model_outputs
    # Check if there are more than one distinct fallacy codes
    return len(distinct_fallacies) > 1

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
                for i in range(len(output)):
                    answers = find_double_answers(output[i])
            if answers:
                print(f"Double answers found in {key}")
def main():
    folder_path = 'results'
    json_files_data = load_all_json_files(folder_path)
    check_double_answers(json_files_data)
    for file_name, data in json_files_data.items():
        print(f'File: {file_name}')


if __name__ == '__main__':
    main()