import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# %matplotlib inline


label_column = 'Class'
bankdata = pd.read_csv("../datasets/bill_authentication.csv")
columns = set(bankdata.columns) - {label_column}
attributes = bankdata.drop(label_column, axis=1)
labels = bankdata[label_column]

# ------------------------------------------------------------------------------ Preprocessing

from sklearn import model_selection
from sklearn.svm import SVC

model = model_selection.train_test_split(attributes, labels, test_size=0.20)
train_attributes, test_attributes, train_labels, test_labels = model

classifier = SVC(kernel='linear')
classifier.fit(train_attributes, train_labels)

# ------------------------------------------------------------------------------- Processing and evaluating

from sklearn.metrics import classification_report, confusion_matrix

predictions = classifier.predict(test_attributes)
# print(confusion_matrix(test_labels, predictions))
# print(classification_report(test_labels, predictions))

# --------------------------------------------------------------------------------- Plot results
import itertools

dataset = test_attributes  # .iloc[:, :2].values  # first try with 2 columns

graph_axes = list(itertools.combinations(columns, 2))
column_indexes = {c: i for c, i in enumerate(columns)}
n = len(graph_axes)
boxes_number = 3
fig, axs = plt.subplots(int(n / boxes_number), boxes_number)
for i, (column_x, column_y) in enumerate(graph_axes):
    r, c = divmod(i, 2)
    g = axs[c, r]
    g.scatter(x=dataset[column_x],
              y=dataset[column_y],
              c=predictions,
              cmap=plt.cm.coolwarm)

    g.set(xlabel=column_x,
          ylabel=column_y)

fig.set_size_inches(w=30, h=20, forward=True)
fig.suptitle('SVC with linear kernel', fontsize=30)
plt.show()

if __name__ == '__main__':
    # --------------------------------------------------------------------------------- Dump and reload
    import pickle

    filename = 'classifier.pkl'
    pickle.dump(classifier, open(filename, 'wb'))
    loaded_classfier = pickle.load(open(filename, 'rb'))
    predictions = loaded_classfier.predict(test_attributes)
    # print(confusion_matrix(test_labels, predictions))
    # print(classification_report(test_labels, predictions))
