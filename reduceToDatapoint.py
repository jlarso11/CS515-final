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
	pull_requests = repo['issues']['pull_requests']

	smallContributors = 0

	for contributor in contributors:
		if contributor['contributions'] < 10:
			smallContributors += 1

	datapoints.append([len(contributors), smallContributors, len(contributors) - smallContributors, pull_requests, defects])


numpyArray = np.array(datapoints)

X = numpyArray[:, :4].reshape(-1,4)

normalized_X = X / np.sqrt(np.sum(X**2))

y = numpyArray[:, 4].reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train) # training the algorithm

y_pred = regressor.predict(X_test)

print(rmse(y_pred, y_test))
print(mse(y_pred, y_test))


X_train, X_test, y_train, y_test = train_test_split(normalized_X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train.ravel())

print(logreg)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
