import json
import os

directory = "results"
new_directory = "results_formatted"

os.makedirs(new_directory, exist_ok=True)

for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), 'r') as old:
        old = json.load(old)

    with open(os.path.join(new_directory, filename), 'w') as new:
        new.write(json.dumps(old, indent=4))