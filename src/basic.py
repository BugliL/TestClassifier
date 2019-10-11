import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# %matplotlib inline


label_column = 'Class'
bankdata = pd.read_csv("../datasets/bill_authentication.csv")
columns = [c for c in bankdata.columns if c != label_column]
attributes = bankdata.drop(label_column, axis=1)
labels = bankdata[label_column]

# ------------------------------------------------------------------------------ Preprocessing

from sklearn import model_selection
from sklearn.svm import SVC

model = model_selection.train_test_split(attributes, labels, test_size=0.20)
train_attributes, test_attributes, train_labels, test_labels = model

classifier = SVC(kernel='rbf')
classifier.fit(train_attributes, train_labels)

# ------------------------------------------------------------------------------- Processing and evaluating

from sklearn.metrics import classification_report, confusion_matrix

predictions = classifier.predict(test_attributes)
# print(confusion_matrix(test_labels, predictions))
# print(classification_report(test_labels, predictions))

# --------------------------------------------------------------------------------- Plot results
import itertools

dataset = test_attributes  # .iloc[:10,:]  # .iloc[:, :2].values  # first try with 2 columns
h = 0.1

graph_axes = list(itertools.combinations(columns, 2))
column_indexes = {c: i for i, c in enumerate(columns)}
column_number = len(column_indexes.items())
n = len(graph_axes)
boxes_number = 3
fig, axs = plt.subplots(int(n / boxes_number), boxes_number)


def create_dataset_from_mesh_grid(xx, yy, columns_x_y):
    (column_x, column_y) = columns_x_y
    raveled_xx, raveled_yy = xx.ravel(), yy.ravel()

    arr = []
    for i in range(column_number):
        if i == columns.index(column_x):
            arr.append(raveled_xx)
        elif i == columns.index(column_y):
            arr.append(raveled_yy)
        else:
            arr.append(np.array([0.0] * len(raveled_xx)))

    return np.c_[arr[0], arr[1], arr[2], arr[3]]


def create_meshed_dataset():
    x_, y_ = dataset.iloc[:, columns.index(column_x)], dataset.iloc[:, columns.index(column_y)]
    x_min, x_max = x_.min() - 1, x_.max() + 1
    y_min, y_max = y_.min() - 1, y_.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy, create_dataset_from_mesh_grid(xx, yy, (column_x, column_y))


for i, (column_x, column_y) in enumerate(graph_axes):
    r, c = divmod(i, 2)
    g = axs[c, r]

    xx, yy, mesh_grid = create_meshed_dataset()
    Z = classifier.predict(mesh_grid).reshape(xx.shape)

    g.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)

    g.scatter(x=dataset[column_x],
              y=dataset[column_y],
              c=predictions,
              cmap=plt.cm.coolwarm)

    g.set(xlabel=column_x,
          ylabel=column_y)

fig.set_size_inches(w=30, h=20, forward=True)
fig.suptitle('SVC with linear kernel', fontsize=30)
plt.savefig("graph.png")
plt.show()

if __name__ == '__main__':
    # --------------------------------------------------------------------------------- Dump and reload
    import pickle
    #
    # filename = 'classifier.pkl'
    # pickle.dump(classifier, open(filename, 'wb'))
    # loaded_classfier = pickle.load(open(filename, 'rb'))
    # predictions = loaded_classfier.predict(test_attributes)
    # print(confusion_matrix(test_labels, predictions))
    # print(classification_report(test_labels, predictions))
