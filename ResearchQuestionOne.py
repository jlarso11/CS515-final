import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from matplotlib import pyplot as plt


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()


# https://stackabuse.com/reading-and-writing-json-to-a-file-in-python/
def readJsonFile(fileName): 
	f = open(fileName)
	with f as json_file:
		return json.load(json_file)
	f.close()


repos = readJsonFile('dataToUse/topData_complete_2')

datapoints = []

for repo in repos :
	defects = repo['issues']['defects']
	contributors = repo['contributors']

	datapoints.append([len(contributors), defects])

numpyArray = np.array(datapoints)

X = numpyArray[:, 0].reshape(-1,1)

y = numpyArray[:, 1].reshape(-1,1)


plt.scatter(X, y)
plt.xlabel("Number of Contributors")
plt.ylabel("Number of Defects")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train) # training the algorithm

y_pred = regressor.predict(X_test)

print(mse(y_pred, y_test))
print(rmse(y_pred, y_test))


X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train.ravel())

print(logreg)
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(y_pred.reshape(-1,1), y_test)))
