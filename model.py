import csv
import numpy
from sklearn.linear_model import LogisticRegression

filename = 'data.csv'

# Lists to store data
y = []
x = []

with open(filename, 'r') as file:
    reader = csv.reader(file)
    
    # Skip the header
    next(reader)
    
    for row in reader:
        y.append(int(row[10]))
        x_row = [float(item) for item in row[1:]]
        x.append(x_row)

y_train = y[300:]
y_test = y[:300]
x_train = x[300:]
x_test = x[:300]
model = LogisticRegression(solver='liblinear', random_state=0).fit(x_train[::-1], y_train[::-1])


predictions = model.predict(x_test)


# Get the predicted probabilities for the test set
proba = model.predict_proba(x_test)
print(proba)
# Set a new threshold
new_threshold = 0.01

# Apply the new threshold
y_pred = numpy.where(proba[:, 1] >= new_threshold, 1, 0)

from sklearn.metrics import confusion_matrix, classification_report

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Compute the classification report
cr = classification_report(y_test, y_pred)

# Print the classification report
print("Classification Report:")
print(cr)

print(model.score(x_test, y_test))




