import json
import os
import glob
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def read_results(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def compute_metrics(true_labels, predicted_labels, classification_level):
    y_true = []
    y_pred = []
    
    for true, pred in zip(true_labels, predicted_labels):
        if classification_level == 0:
            true = 'fallacy' if any(label != 'no fallacy' for label in true) else 'no fallacy'
            pred = 'fallacy' if pred != 'no fallacy' else 'no fallacy'
        if classification_level == 1:
            pass 
        else:
            pass 
        y_true.append(true)
        y_pred.append(pred)
    
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    
    return precision, recall, f1, accuracy


def main(folder_path: str):
    
    file_list = glob.glob(os.path.join(folder_path, '*.json'))
    for file_path in file_list:
        print(file_path)
        results = read_results(file_path)
        
        # Extract true and predicted labels
        true_labels = [results[str(i)]['true_labels'] for i in range(1, results['n_samples'] + 1)]
        predicted_labels = [results[str(i)]['predicted_fallacies'] for i in range(1, results['n_samples'] + 1)]
        classification_level = results['level']  

        precision, recall, f1, accuracy = compute_metrics(true_labels, predicted_labels, classification_level)
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    folder_path = 'results'
    main(folder_path)
