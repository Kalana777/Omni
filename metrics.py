import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(all_predictions, all_labels):

    predicted_labels = torch.argmax(all_predictions, dim=1).cpu().numpy()
    true_labels = all_labels.cpu().numpy()

    accuracy = accuracy_score(true_labels, predicted_labels)

    precision = precision_score(true_labels, predicted_labels)

    recall = recall_score(true_labels, predicted_labels)

    f1 = f1_score(true_labels, predicted_labels)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_metrics2(all_predictions, all_labels):
    tot_p = 0
    true_p = 0
    pred_p = 0

    # y_pred = all_predictions.squeeze(0).tolist()
    y_pred = torch.argmax(all_predictions, dim=1).cpu().squeeze(0).tolist()
    valid_y_tensor = all_labels.squeeze(0).tolist()

    for i in range(len(y_pred)):

        if valid_y_tensor[i] > 0:
            tot_p += 1

            if y_pred[i] == valid_y_tensor[i]:
                true_p += 1

        if y_pred[i] > 0:
            pred_p += 1

    f1 = 0.0
    prec = 0.0
    rec = 0.0

    if tot_p and pred_p:
        rec = true_p / tot_p
        prec = true_p / pred_p

        if rec > 0 or prec > 0:
            f1 = 2 * prec * rec / (prec + rec)

    print('P: ' + str(round(prec, 4)) + '  |  R: ' + str(round(rec, 4)) + '  |  F1: ' + str(round(f1, 4)))

    return {
        'accuracy': 1,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }