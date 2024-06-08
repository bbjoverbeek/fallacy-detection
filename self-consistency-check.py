import json
import os
import glob
import argparse
import pandas as pd
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
def read_results(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_most_frequent_label(predicted_list, all_labels):
    """identifies most frequent label in the list"""
    if not predicted_list:
        return "no fallacy"
    counter = Counter(predicted_list)
    most_common = counter.most_common(1)
    label = most_common[0][0] if most_common else "no fallacy"
    return label if label in all_labels else "no fallacy"

def get_predicted_labels(predicted_fallacies, prompt_features, all_labels):
    """for self consistency, it choose the most frequent label,
    otherwise it uses the first label"""

    if 'self-consistency' in prompt_features:
        consistent_count = 0

        for predictions in predicted_fallacies:
            # Check if all elements in the predictions list are the same
            if len(set(predictions)) == 1:
                consistent_count += 1

        return consistent_count
        

def flatten_labels(labels, classification_level):
    if classification_level == 0:
        return ['fallacy' if any(label != 'no fallacy' for label in lbl) else 'no fallacy' for lbl in labels]
    else:
        return [label for sublist in labels for label in sublist]

def check_labels_consistency(list_of_labels, level, all_labels):
    curted_list = []
    for labels in list_of_labels:
        temp_list=[]
        for label in labels:
            if label in all_labels:
                temp_list.append(label)
        if len(temp_list)==0:
            temp_list.append('no fallacy')
        curted_list.append(temp_list)
    return curted_list

def compute_metrics(true_labels, predicted_labels, level, all_labels):
    y_true = true_labels
    y_pred = predicted_labels
    if level ==1 or level==2:
        # y_true = check_labels_consistency(true_labels, level, all_labels)
        # y_pred = check_labels_consistency(predicted_labels, level, all_labels)
        mlb = MultiLabelBinarizer()
        y_true = mlb.fit_transform(y_true)
        y_pred = mlb.transform(y_pred)
        precision = precision_score(y_true, y_pred, average="micro")
        recall = recall_score(y_true, y_pred,  average="micro")
        f1 = f1_score(y_true, y_pred, average="micro")
    else:
        y_true = [x[0] for x in y_true]
        precision = precision_score(y_true, y_pred, pos_label="fallacy")
        recall = recall_score(y_true, y_pred, pos_label="fallacy")
        f1 = f1_score(y_true, y_pred, pos_label="fallacy")

    accuracy = accuracy_score(y_true, y_pred)    
    
    return precision, recall, f1, accuracy

def process_file(file_path):
    data = read_results(file_path)

    classification_level = data['level']

    all_labels = set(pd.read_json('classification_level.jsonl', lines=True)['labels'][classification_level])

    true_labels = [data[str(i)]['true_labels'] for i in range(1, data['n_samples'] + 1)]
    predicted_fallacies = [data[str(i)]['predicted_fallacies'] for i in range(1, data['n_samples'] + 1)]
    prompt_features = data['prompt_features']
    if 'self-consistency' in data['prompt_features'][0]:
        print(f"-----{file_path}---")
        predicted_labels = get_predicted_labels(predicted_fallacies, prompt_features, all_labels)
        print(f"Level: {classification_level}\nRepeat: {data['self-consistency repetitions']}")
        print(f"Obtained: {predicted_labels} times same response")
    
	


    

def main():
    parser = argparse.ArgumentParser(description="Evaluate fallacy detection model")
    parser.add_argument('-i', '--folder_path', default="results", type=str, help='Path to the input JSON file')
    #parser.add_argument('-l', '--labels_file', type=str, required=True, help='Path to the file containing all possible labels')
    args = parser.parse_args()

    

    file_list = glob.glob(os.path.join(args.folder_path, '*.json'))
    for file_path in file_list:
        
        result = process_file(file_path)
        
        

if __name__ == '__main__':
    main()
