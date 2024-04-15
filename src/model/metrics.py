
import numpy as np
from transformers import EvalPrediction
from sklearn.metrics import balanced_accuracy_score,f1_score,recall_score

is_regression=False

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels,preds,average='weighted')
    recall = recall_score(labels,preds,average='weighted')
    accuracy = balanced_accuracy_score(labels,preds)
    return {"accuracy": accuracy, "f1": f1, "recall": recall}
