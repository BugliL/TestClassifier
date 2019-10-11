import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline


bankdata = pd.read_csv("../datasets/bill_authentication.csv")
attributes = bankdata.drop('Class', axis=1)
labels = bankdata['Class']

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
print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))

if __name__ == '__main__':
    import pickle

    filename = 'classifier.pkl'
    pickle.dump(classifier, open(filename, 'wb'))
    loaded_classfier = pickle.load(open(filename, 'rb'))
    predictions = loaded_classfier.predict(test_attributes)
    print(confusion_matrix(test_labels, predictions))
    print(classification_report(test_labels, predictions))
