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

# def get_most_frequent_label(predicted_list, all_labels):
#     """identifies most frequent label in the list"""
#     if not predicted_list:
#         return "no fallacy"
#     counter = Counter(predicted_list)
#     most_common = counter.most_common(1)
#     label = most_common[0][0] if most_common else "no fallacy"
#     return label if label in all_labels else "no fallacy"

# def get_predicted_labels(predicted_fallacies, prompt_features, all_labels):
#     """for self consistency, it choose the most frequent label,
#     otherwise it uses the first label"""

#     if 'self-consistency' in prompt_features:
#         return [get_most_frequent_label(predictions, all_labels) if isinstance(predictions, list) else (predictions if predictions in all_labels else "no fallacy") for predictions in predicted_fallacies]
#     else:
#         return [predictions[0] if predictions[0] in all_labels else "no fallacy" for predictions in predicted_fallacies]

def get_most_frequent_label(predicted_list, all_labels):
    """Identifies the most frequent label in the list."""
    # Check if the list of predictions is empty
    false_label =0
    if not predicted_list:
        false_label =1
        return "no fallacy", false_label

    # Count the occurrences of each label in the list
    counter = Counter(predicted_list)

    # Get the most common label and its count
    most_common = counter.most_common(1)

    # Check if there is a most common label
    if most_common:
        label = most_common[0][0]  # Get the label with the highest count
    else:
        label = "no fallacy"  # Default to "no fallacy" if no common label is found

    # Ensure the most common label is a valid label
    if label in all_labels:
        return label, false_label
    else:
        false_label =1
        return "no fallacy", false_label


def get_predicted_labels(predicted_fallacies, prompt_features, all_labels):
    """For self-consistency, choose the most frequent label; otherwise, use the first label."""

    predicted_labels = []
    total_invalid_output =0
    # Check if 'self-consistency' is one of the prompt features
    if 'self-consistency' in prompt_features:
        # Process each set of predictions
        for predictions in predicted_fallacies:
            # Check if predictions are in a list (self-consistency case)
            if isinstance(predictions, list):
                # Get the most frequent label from the list of predictions
                most_frequent_label, invalid_label = get_most_frequent_label(predictions, all_labels)
                predicted_labels.append(most_frequent_label)
                if invalid_label:
                    total_invalid_output+=1
            else:
                # Handle case where predictions are not in list form (edge case)
                if predictions in all_labels:
                    predicted_labels.append(predictions)
                else:
                    total_invalid_output +=1
                    predicted_labels.append("no fallacy")
    else:
        # Process each set of predictions for other prompt features
        for predictions in predicted_fallacies:
            # Use the first prediction if it's a valid label
            if predictions[0] in all_labels:
                predicted_labels.append(predictions[0])
            else:
                total_invalid_output +=1
                predicted_labels.append("no fallacy")

    return predicted_labels, total_invalid_output



def compute_metrics(true_labels, predicted_labels, level, all_labels):
    y_true = true_labels
    y_pred = predicted_labels
    if level ==1 or level==2:
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

    predicted_labels, total_invalid_output = get_predicted_labels(predicted_fallacies, prompt_features, all_labels)
    if classification_level ==1 or classification_level==2:
        predicted_labels= [[label] for label in predicted_labels]
        unique_true_labels = []
        for labels in true_labels:
            unique_true_labels.append(list(set(labels)))
        true_labels = unique_true_labels
    precision, recall, f1, accuracy = compute_metrics(true_labels, predicted_labels, classification_level, all_labels)
    if (len(data['prompt_features'])>1):
        prompt_features = data['prompt_features'][0] +"-"+ data['prompt_features'][1]
    else: 
        prompt_features = data['prompt_features'][0]
    return {
    'file': file_path,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'accuracy': accuracy,
    'model' : data['model'],
    'prompt_feature': prompt_features,
    'n_shot': data['n_shot'],
    'repeat': data['self-consistency repetitions'],
    'level': classification_level,
    'invalid_output': total_invalid_output
    }
	


    

def main():
    parser = argparse.ArgumentParser(description="Evaluate fallacy detection model")
    parser.add_argument('-i', '--folder_path', default="results", type=str, help='Path to the input JSON file')
    #parser.add_argument('-l', '--labels_file', type=str, required=True, help='Path to the file containing all possible labels')
    args = parser.parse_args()

    

    file_list = glob.glob(os.path.join(args.folder_path, '*.json'))
    all_models = []
    all_prompt_features = []
    all_n_shot =[]
    all_repeat = []
    all_acc = []
    all_pre = []
    all_rec = []
    all_f1 =[]
    all_levels=[]
    invalid_output=[]
    for file_path in file_list:
        print(f"------{file_path}----------")
        result = process_file(file_path)
        print(f"File: {result['file']}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"F1 Score: {result['f1_score']:.4f}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("---------------------------------")
        all_models.append(result['model'])
        all_prompt_features.append(result['prompt_feature'])
        all_n_shot.append(result['n_shot'])
        all_repeat.append(result['repeat'])
        all_acc.append(result['accuracy'])
        all_pre.append(result['precision'])
        all_rec.append(result['recall'])
        all_f1.append(result['f1_score'])
        all_levels.append(result['level'])
        invalid_output.append(result['invalid_output'])
    df = pd.DataFrame({"model": all_models, 'level':all_levels, "prompt_feature": all_prompt_features, "accuracy": all_acc, "f1_score":all_f1, "precision": all_pre, "recall": all_rec, "n_shot":all_n_shot, "repeat": all_repeat, "invalid_output":invalid_output})
    df.to_csv("output_micro_scores_2.csv")

if __name__ == '__main__':
    main()
