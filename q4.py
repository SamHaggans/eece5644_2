import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.naive_bayes import MultinomialNB

np.random.seed(35235)

X_all = []
Y_all = []

words = []

with open("words.txt", "r") as wordsfile:
    for line in wordsfile:
        words = line.strip().split(",")

words = np.array(words)

with open("X_snts.csv", "r") as csvfile:
    for line in csvfile:
        X_all.append([int(a) for a in line.strip().split(",")])

with open("y_snts.csv", "r") as csvfile:
    for line in csvfile:
        Y_all.append([int(a) for a in line.strip().split(",")])

X_all_np = np.array(X_all)
Y_all_np = np.array(Y_all).ravel()
sample_count = len(X_all_np)


TRAIN_PERCENT = 80

training_count = int((TRAIN_PERCENT / 100) * sample_count)


alphas = np.logspace(-15, 5, num=40, base=2)

accuracy_means = []
accuracy_stdevs = []
categorizers = []
for alpha in alphas:
    accuracies = []
    _categorizers = []
    for _ in range(10):
        training_indices = np.random.choice(sample_count, size=training_count, replace=False)
        X_train = X_all_np[training_indices]
        Y_train = Y_all_np[training_indices]
        test_indices = np.setdiff1d(np.arange(sample_count), training_indices)
        X_test = X_all_np[test_indices]
        Y_test = Y_all_np[test_indices]
        categorizer = MultinomialNB(alpha=alpha)
        categorizer.fit(X_train, Y_train)
        _categorizers.append(categorizer)
        predicted_y = categorizer.predict(X_test)
        accuracy = categorizer.score(X_test, Y_test)
        accuracies.append(accuracy)
    accuracy_means.append(np.mean(accuracies))
    accuracy_stdevs.append(np.std(accuracies))
    categorizers.append(_categorizers[0])
best_index = np.argmax(accuracy_means)
print("Best Alpha: ", alphas[best_index])
print("Best Accuracy: ", accuracy_means[best_index])
cats = ["MISC", "AIMX", "OWNX", "CONT", "BASE"]
for i, cl in enumerate(categorizers[best_index].feature_log_prob_):
    sorted_indices = np.argsort(cl)
    largest_indices = sorted_indices[-5:]
    print(f"Highest paramter value words for {cats[i]}: {words[largest_indices]}")
accuracy_means = np.array(accuracy_means)
accuracy_stdevs = np.array(accuracy_stdevs)

plt.figure(figsize=(8, 8))
plt.plot(alphas, accuracy_means, color="darkorange", lw=2, label="Accuracy")
plt.fill_between(alphas, accuracy_means+accuracy_stdevs, accuracy_means-accuracy_stdevs, facecolor='C0', alpha=0.4)
plt.xlabel("Alpha")
plt.xscale("log")
plt.ylabel("Score")
plt.title(f"Accuracy vs Alpha")
plt.legend(loc="lower right")
plt.show()
