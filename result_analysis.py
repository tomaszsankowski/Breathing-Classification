from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import tensorflow as tf


def analyse_error(clf, x_train, y_train, decision_function=False):
    num_class = 3
    # Calculating cross_validation predictions
    y_train_predict = cross_val_predict(clf, x_train, y_train, cv=3)

    # Creating confusion matrix
    c_matrix = confusion_matrix(y_train, y_train_predict, labels=[0,1,2])
    print("\n* * *\n")
    print("Confusion matrix: ")
    print(c_matrix)
    print("Rows - Actual values\nColumns - predicted values")
    print("Classes: 0 - inhale, 1 - exhale, 2 - silence")

    # Calculating precision and recall
    calculate_precision_and_recall(clf, x_train, y_train, y_train_predict)

    #Some regressors (like random forest) do not have decision function
    #In those cases use probability
    if decision_function:
        y_scores = cross_val_predict(clf, x_train, y_train, cv=3, method="decision_function")
    else:
        y_probs = cross_val_predict(clf, x_train, y_train, cv=3, method="predict_proba")
        precisions, recalls, thresholds = [0,0,0],[0,0,0],[0,0,0]
        fpr, tpr, thresholds_roc = [0,0,0],[0,0,0],[0,0,0]
        auc = [0,0,0]
        for i in range(num_class):
            y_scores = y_probs[:, i]

            #Calculating precisions and recalls with various thresholds for each class
            precisions[i], recalls[i], thresholds[i] = precision_recall_curve(y_true=tf.equal(y_train, i),
                                                                     probas_pred=y_scores)

            #Calculating false positive rate and true positive rate with various thresholds for each class
            fpr[i], tpr[i], thresholds[i] = roc_curve(y_true=tf.equal(y_train, i), y_score=y_scores)
            auc[i] = roc_auc_score(y_true=tf.equal(y_train, i), y_score=y_scores)

        #Plotting precision in a function of recall
        plot_precision_vs_recall(precisions, recalls, class_num=num_class)
        plt.show()

        #Ploting Receiver Operating Characteristic curve
        #(true positive rate in a function of false positive rate)
        plot_roc_curve(fpr, tpr, class_num=num_class)
        plt.show()

        # AUC - area under the ROC curve
        print("\n* * *\n")
        for i in range(3):
            class_name = "inhale" if i == 0 else ("exhale" if i == 1 else "silence")
            print("AUC for ", i," - ", class_name,": ", auc[i])

    # Plotting precision and recall in a function of threshold
    #precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    #plot_precision_recall_vs_threshold(precisions, recalls, thresholds)


#calculatng precision and recall for each class
#calculating f1 as a measure combining precision and recall
#precision = TP/(TP+FP)                (T-True, F-False, P-Positive, N-Negative)
#recall = TP/(TP+FN)
#f1 = 2*(precision*recall/(precision+recall))
def calculate_precision_and_recall(clf, x_train, y_train, y_pred):
    precisions, recalls, f1_scores = [],[],[]
    for i in range(3):
        precisions.append(precision_score(y_true=tf.equal(y_train, i), y_pred=tf.equal(y_pred,i)))
        recalls.append(recall_score(y_true=tf.equal(y_train, i), y_pred=tf.equal(y_pred,i)))
        f1_scores.append(f1_score(y_true=tf.equal(y_train, i), y_pred=tf.equal(y_pred,i)))

    print("\n* * *\n")
    for i in range(3):
        class_name = "inhale" if i == 0 else ("exhale" if i== 1 else "silence")
        print("\nClass ", i, " - ", class_name)
        print("Precision: ", precisions[i])
        print("Recall: ", recalls[i])
        print("F1 score: ", f1_scores[i])


# Plotting functions
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    #plt.plot(recalls, precisions, "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])
    plt.show()

def plot_precision_vs_recall(precisions, recalls, class_num):
    for i in range(class_num):
        class_name = "inhale" if i == 0 else ("exhale" if i == 1 else "silence")
        plt.plot(recalls[i], precisions[i], "--", label=str(i)+ ":" +class_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="center left")
    plt.title("Precision vs recall curve")
    plt.ylim([0, 1])

def plot_roc_curve(fpr, tpr, class_num, label=None):
    for i in range(class_num):
        class_name = "inhale" if i == 0 else ("exhale" if i == 1 else "silence")
        plt.plot(fpr[i], tpr[i], linewidth=2, label=str(i)+ ":" +class_name)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis((0.0, 1.0, 0.0, 1.0))
    plt.legend(loc="center left")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
