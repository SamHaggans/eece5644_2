# Example code to load a dataset, train a GaussianNaiveBayes model, and test it
# See:
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html 
# https://scikit-learn.org/stable/modules/naive_bayes.html
# https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,confusion_matrix,roc_curve
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


X, y = datasets.load_breast_cancer(return_X_y=True)

print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))



classifier = GaussianNB()

print("Training classifier...",end="")
classifier.fit(X_train, y_train)
print(" done")


# Compute all kinds of prediction performance metrics on the test set

#get preditions on test set
y_pred =  classifier.predict(X_test) 

acc = accuracy_score(y_test,y_pred)   #ACC = (TP + TN) / (P+N)
print("Accuracy:",acc)

f1 = f1_score(y_test,y_pred)    #F1 = 2 * TP / (2 * TP + FN + FP)
print("F1:",f1)


cm = confusion_matrix(y_test,y_pred)
print("Confusion matrix:\n",cm)




#get prob scores (discriminant function values of each class)
probs = classifier.predict_proba(X_test)

fpr,tpr,thresholds = roc_curve(y_test,probs[:,1])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(fpr,tpr)
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
fig.savefig("ROC.pdf", bbox_inches="tight")

auc = roc_auc_score(y_test,probs[:,1])
print("AUC:",auc)





