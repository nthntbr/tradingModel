import numpy
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pandas.read_csv('data.csv')

# Separate features and target variable from the DataFrame
df = df.iloc[16:-1]

X = df.drop('Target', axis=1)
X = X.drop('Next_Day_Return', axis=1)
X = X.drop('1. open', axis=1)
X = X.drop('2. high', axis=1)
X = X.drop('3. low', axis=1)
X = X.drop('Return', axis=1)
y = df['Target']
# Split the data into training and testing sets (70% train, 30% test in this case)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.075, random_state=42)


model = LogisticRegression(solver='liblinear', random_state=0).fit(X_train[::-1], y_train[::-1])


predictions = model.predict(X_test)


# Get the predicted probabilities for the test set
proba = model.predict_proba(X_test)
print(proba)
# Set a new threshold
new_threshold = 0.5

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

print(model.score(X_test, y_test))




