from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


def analyse_error(clf, x_train, y_train, decision_function=False):
    # Calculation cross_validation predictions
    y_train_predict = cross_val_predict(clf, x_train, y_train, cv=3)

    # Creating confusion matrix
    c_matrix = confusion_matrix(y_train, y_train_predict)
    print("Confusion matrix: ")
    print(c_matrix)

    # Calculating precision and recall
    calculate_precision_and_recall(clf, x_train, y_train)

    # Some regressors (like random forest) do not have decision function
    # In those cases we use probability
    if decision_function:
        y_scores = cross_val_predict(clf, x_train, y_train, cv=3, method="decision_function")
    else:
        y_probs = cross_val_predict(clf, x_train, y_train, cv=3, method="predict_proba")
        y_scores = y_probs[:, 1]

    # Plotting precision and recall in a function of threshold
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

    # Receiver Operating Characteristic curve plotting
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)
    plot_roc_curve(fpr, tpr)

    # AUC - area under the ROC curve
    print("AUC:", roc_auc_score(y_train, y_scores))


# Printing precision, recall and F1 score
def calculate_precision_and_recall(clf, x_train, y_train):
    y_train_predict = cross_val_predict(clf, x_train, y_train, cv=3)
    precision = precision_score(y_train, y_train_predict)
    recall = recall_score(y_train, y_train_predict)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1:", f1_score(y_train, y_train_predict))


# Plotting functions
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    plt.show()


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.axis((0.0, 1.0, 0.0, 1.0))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.show()
