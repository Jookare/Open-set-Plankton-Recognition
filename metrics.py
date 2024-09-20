import numpy as np

def get_mask(y, num_train_classes):
    mask_known = y < num_train_classes
    mask_unk = y >= num_train_classes
    return mask_known, mask_unk


# Calculates the open-set accuracy
def os_accuracy(y, y_pred):
    total = len(y)
    acc = np.sum(y_pred == y) / total
    return acc

def calc_accuracies(labels, predictions, num_train_classes):
    mask_known, mask_unk = get_mask(labels, num_train_classes)
    known_accuracy = np.mean(predictions[mask_known] == labels[mask_known])
    unknown_accuracy = np.mean(predictions[mask_unk] == num_train_classes)
    print(
        "Known classes={:.4f}, Unknown classes={:.4f}, N-Known images {}, N-Unknown images {}".format(
            known_accuracy,
            unknown_accuracy,
            np.sum(mask_known),
            np.sum(mask_unk)
        )
    )
    return known_accuracy, unknown_accuracy


def perf_measure(y_actual, y_pred):
    class_id = sorted(set(y_actual).union(set(y_pred)))
    TP = [0] * len(class_id)
    FP = [0] * len(class_id)
    TN = [0] * len(class_id)
    FN = [0] * len(class_id)
    for _id in class_id:
        for i in range(len(y_pred)):
            if y_actual[i] == _id and y_pred[i] == _id:
                TP[_id] += 1
            if y_pred[i] == _id and y_actual[i] != _id:
                FP[_id] += 1
            if y_pred[i] != _id and y_actual[i] != _id:
                TN[_id] += 1
            if y_pred[i] != _id and y_actual[i] == _id:
                FN[_id] += 1

    return class_id, TP, FP, TN, FN


def compute_os_f1(label, pred):
    """Label are correct labels, such that the unknown classes are considered to be one class.
    predictions are predicted labels after the thresholding
    """

    res = perf_measure(label, pred)
    recall_micro = np.sum(res[1][:-1]) / np.sum(res[1][:-1] + res[4][:-1])
    precision_micro = np.sum(res[1][:-1]) / np.sum(res[1][:-1] + res[2][:-1])

    f_measure = np.round(2 * precision_micro * recall_micro / (precision_micro + recall_micro), 4)
    f_measure = f_measure
    return f_measure


def compute_metrics(labels_test, preds_no_th, preds, num_classes):
    """
    Args:
        * labels_test (numpy arr) Correct labels of test set
        * preds_no_th (numpy arr) Predicted labels without thresholding
        * preds (numpy arr) Predicted labels with thresholding
        * num_classes (N) number of known classes in training set
    """
    # Transfrom all unknown classes to same label
    _, mask_unk = get_mask(labels_test, num_classes)
    labels_test[mask_unk] = num_classes

    # Calculate accuracies
    acc_known, _ = calc_accuracies(labels_test, preds_no_th, num_classes)
    acc_known_th, acc_unk_th = calc_accuracies(labels_test, preds, num_classes)

    # Calculate open-set accuracy and open-set f-measure
    acc_os = os_accuracy(labels_test, preds)
    f1_os = compute_os_f1(labels_test, preds)

    return {"acc_known": acc_known, "acc_known_th": acc_known_th, "acc_unk_th": acc_unk_th, "acc_os": acc_os, "f1_os": f1_os}
