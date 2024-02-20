import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.naive_bayes import CategoricalNB

np.random.seed(40)

X_all = []
Y_all = []

features_dict = dict()
features_list = []
# Captured col_maps from process_mushroom.py
# Used to get readable names for all of the labels
feature_index_map = {
    0: {"p": 0, "e": 1},
    1: {"x": 0, "b": 1, "s": 2, "f": 3, "k": 4, "c": 5},
    2: {"s": 0, "y": 1, "f": 2, "g": 3},
    3: {"n": 0, "y": 1, "w": 2, "g": 3, "e": 4, "p": 5, "b": 6, "u": 7, "c": 8, "r": 9},
    4: {"t": 0, "f": 1},
    5: {"p": 0, "a": 1, "l": 2, "n": 3, "f": 4, "c": 5, "y": 6, "s": 7, "m": 8},
    6: {"f": 0, "a": 1},
    7: {"c": 0, "w": 1},
    8: {"n": 0, "b": 1},
    9: {
        "k": 0,
        "n": 1,
        "g": 2,
        "p": 3,
        "w": 4,
        "h": 5,
        "u": 6,
        "e": 7,
        "b": 8,
        "r": 9,
        "y": 10,
        "o": 11,
    },
    10: {"e": 0, "t": 1},
    11: {"e": 0, "c": 1, "b": 2, "r": 3, "?": 4},
    12: {"s": 0, "f": 1, "k": 2, "y": 3},
    13: {"s": 0, "f": 1, "y": 2, "k": 3},
    14: {"w": 0, "g": 1, "p": 2, "n": 3, "b": 4, "e": 5, "o": 6, "c": 7, "y": 8},
    15: {"w": 0, "p": 1, "g": 2, "b": 3, "n": 4, "e": 5, "y": 6, "o": 7, "c": 8},
    16: {"p": 0},
    17: {"w": 0, "n": 1, "o": 2, "y": 3},
    18: {"o": 0, "t": 1, "n": 2},
    19: {"p": 0, "e": 1, "l": 2, "f": 3, "n": 4},
    20: {"k": 0, "n": 1, "u": 2, "h": 3, "w": 4, "r": 5, "o": 6, "y": 7, "b": 8},
    21: {"s": 0, "n": 1, "a": 2, "v": 3, "y": 4, "c": 5},
    22: {"u": 0, "g": 1, "m": 2, "d": 3, "p": 4, "w": 5, "l": 6},
}

# Reads attributes to convert the above single-character labels into readable text
with open("mushroom/attrs.txt", "r") as attrs_file:
    for line in attrs_file:
        prop = line.split(":")[0]
        prop_values = dict()
        features_list.append(prop)
        for element in line.split(":")[1].strip().split(","):
            k = element.split("=")[1]
            text = element.split("=")[0]
            prop_values[k] = text
        features_dict[prop] = prop_values

with open("X_msrm.csv", "r") as csvfile:
    for line in csvfile:
        X_all.append([int(a) for a in line.strip().split(",")])

with open("y_msrm.csv", "r") as csvfile:
    for line in csvfile:
        Y_all.append([int(a) for a in line.strip().split(",")])

X_all_np = np.array(X_all)
Y_all_np = np.array(Y_all).ravel()
sample_count = len(X_all_np)


TRAIN_PERCENT = 80

training_count = int((TRAIN_PERCENT / 100) * sample_count)

training_indices = np.random.choice(sample_count, size=training_count, replace=False)
X_train = X_all_np[training_indices]
Y_train = Y_all_np[training_indices]
test_indices = np.setdiff1d(np.arange(sample_count), training_indices)
X_test = X_all_np[test_indices]
Y_test = Y_all_np[test_indices]

# From process_mushroom.py, how many categories are expected for each feature.
expected_category_len = [
    6,
    4,
    10,
    2,
    9,
    2,
    2,
    2,
    12,
    2,
    5,
    4,
    4,
    9,
    9,
    1,
    4,
    3,
    5,
    9,
    6,
    7,
]

alphas = np.logspace(-15, 5, num=50, base=2)
accuracies = []
roc_aucs = []
f1s = []
categorizers = []

for alpha in alphas:
    # print(alpha)
    categorizer = CategoricalNB(alpha=alpha, min_categories=expected_category_len)
    categorizer.fit(X_train, Y_train)
    categorizers.append(categorizer)
    predicted_y = categorizer.predict(X_test)
    accuracy = categorizer.score(X_test, Y_test)
    # print("Accuracy Score: ", accuracy)
    predicted_probabilities = categorizer.predict_proba(X_test)
    y_pred_proba_positive_class = predicted_probabilities[:, 1]
    # print(y_pred_proba_positive_class)
    roc_auc = roc_auc_score(Y_test, y_pred_proba_positive_class)
    # print("ROC AUC Score: ", roc_auc)
    f1 = f1_score(Y_test, predicted_y, average="binary")
    # print("F1: ", f1)
    accuracies.append(accuracy)
    roc_aucs.append(roc_auc)
    f1s.append(f1)


best_index = np.argmax(roc_aucs)

# Print out the parameter table in csv format
classes = ["Poisonous", "Edible"]
for i, feature in enumerate(categorizers[best_index].feature_log_prob_):

    print(f"Weights of {features_list[i]}", end=",")    
    index_map = feature_index_map[i + 1]
    for elem in index_map:
        print(features_dict[features_list[i]][elem], end=",")
    print()    
    for c, cls in enumerate(classes):
        feature_weights = feature[c]
        print(f"{cls},", end="")
        for val in feature_weights:
            print(val, end=",")
        print()
        

print(f"Using {TRAIN_PERCENT}% as training data")
print("Best Alpha (Max ROC AUC): ", alphas[best_index])
print("Accuracy: ", accuracies[best_index])
print("ROC AUC: ", roc_aucs[best_index])
print("F1 Score: ", f1s[best_index])

plt.figure(figsize=(8, 8))
plt.plot(alphas, roc_aucs, color="blue", lw=2, label="ROC AUC")
plt.plot(alphas, accuracies, color="darkorange", lw=2, label="Accuracy")
plt.plot(alphas, f1s, color="green", lw=2, label="F1 Score")
plt.xlabel("Alpha")
plt.xscale("log")
plt.ylabel("Score")
plt.title(f"Predictive Performance vs Alpha: {TRAIN_PERCENT}% Train")
plt.legend(loc="lower right")
plt.show()
