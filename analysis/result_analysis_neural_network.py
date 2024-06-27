from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
import tensorflow as tf


def result_analysis_nn(raw_y_pred, Y_test):
  #loss, accuracy = model.evaluate(X_test, Y_test)
  #print(f'Test loss: {loss}, Test accuracy: {accuracy}')
  Y_test_predict_raw = raw_y_pred#model.predict(X_test)

  #Y_test_predict = [(prob.tolist()).index(max(prob.tolist())) for prob in Y_test_predict_raw]
  Y_test_predict = [(prob).index(max(prob)) for prob in Y_test_predict_raw]
  Y_test_array = [(arr).index(1) for arr in Y_test]

  accuracy = calculate_accuracy(Y_test_array, Y_test_predict)
  print(f'Test accuracy: {accuracy}')

  calculate_precision_and_recall(Y_test_array, Y_test_predict)
  plot_precision_vs_recall(Y_test_array, np.array(Y_test_predict_raw), 3)
  plt.show()
  plot_roc(Y_test_array, np.array(Y_test_predict_raw), 3)
  plt.show()

def calculate_precision_and_recall(y_test, y_pred):
    precisions, recalls, f1_scores = [],[],[]
    for i in range(3):
        precisions.append(precision_score(y_true=tf.equal(y_test, i), y_pred=tf.equal(y_pred,i)))
        recalls.append(recall_score(y_true=tf.equal(y_test, i), y_pred=tf.equal(y_pred,i)))
        f1_scores.append(f1_score(y_true=tf.equal(y_test, i), y_pred=tf.equal(y_pred,i)))


    print("\n* * *\n")
    for i in range(3):
        class_name = "inhale" if i == 0 else ("exhale" if i== 1 else "silence")
        print("\nClass ", i, " - ", class_name)
        print("Precision: ", precisions[i])
        print("Recall: ", recalls[i])
        print("F1 score: ", f1_scores[i])

    macro_avg(precisions, recalls)

def macro_avg(precisions, recalls):
    N = len(precisions)
    sum_p, sum_r = 0,0
    for i in range(N):
        sum_p += precisions[i]
        sum_r += recalls[i]
    macro_avg_p = sum_p/N
    macro_avg_r = sum_r/N
    print("\n\nMacro average precison: ", macro_avg_p)
    print("Macro average recall: ", macro_avg_r)

def plot_precision_vs_recall(Y_true, y_probas, classes_num):
    plt.clf()
    precisions, recalls, thresholds = [0,0,0],[0,0,0],[0,0,0]
    for i in range(classes_num):
      y_scores = y_probas[:, i]
      precisions[i], recalls[i], thresholds[i] = precision_recall_curve(y_true=tf.equal(Y_true, i),
                                                                      probas_pred=y_scores)
      class_name = "inhale" if i == 0 else ("exhale" if i == 1 else "silence")
      plt.plot(recalls[i], precisions[i], "--", label=str(i)+ ":" +class_name)
      #plt.plot(recalls, precisions, "b--", label="Precision")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="center left")
    plt.title("Precision vs recall curve")
    plt.ylim([0, 1.01])

def plot_roc(Y_test, Y_probas, class_num):
  plt.clf()
  fpr,tpr,thresholds = [0,0,0],[0,0,0],[0,0,0]
  auc = [0,0,0]

  class_num = 3
  for i in range(class_num):
    y_scores = Y_probas[:, i]
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_true=tf.equal(Y_test, i), y_score=y_scores)
    auc[i] = roc_auc_score(y_true=tf.equal(Y_test, i), y_score=y_scores)
    class_name = "inhale" if i == 0 else ("exhale" if i == 1 else "silence")
    print("AUC for ", i," - ", class_name,": ", auc[i])
    plt.plot(fpr[i], tpr[i], linewidth=2, label=str(i)+ ":" +class_name)


  plt.plot([0, 1.0], [0, 1.0], 'k--')
  plt.axis((0.0, 1.01, 0.0, 1.01))
  plt.legend(loc="center left")
  plt.xlabel("False positive rate")
  plt.ylabel("True positive rate")
  plt.title("ROC curve")

def calculate_accuracy(Y_true, Y_pred):
    right = 0;
    for idc in range(len(Y_true)):
        if(Y_true[idc] == Y_pred[idc]):
            right+=1
    return right/len(Y_true)


confusion_matrix = [[75,31,0],[15,92,1],[15,2,140]]

raw_prediction = []
for i in range(3):
    for j in range(3):
        input = [0,0,0]
        input[j] = 1
        for k in range(confusion_matrix[i][j]):
            raw_prediction.append(input)



def start_analysis(predictions, confusion_matrix):
    raw_res = []
    for p in predictions:
        raw_res.append(p[0].tolist())
    #print(raw_res)

    y_test = []
    for i in range(3):
        for j in range(3):
            for k in range(int(confusion_matrix[i][j])):
                input = [0, 0, 0]
                input[0 if i == 1 else 1 if i == 0 else 2] = 1
                y_test.append(input)
    result_analysis_nn(raw_res, y_test)

