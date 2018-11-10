

# you should first install the xgboost package
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")

X = dataset[:, 0:8]
Y = dataset[;, 8]
# split data into train & test sets
# seed: anchor the random seed so that can lock those
# 		random numbers each iteration
seed = 7
test_size = 0.33

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
	test_size = test_size, random_state = seed)

# fit
model = XGBClassifier()
eval_set = [(X_test, Y_test)]
model.fit(X_train, Y_train, early_stopping_rounds = 10, 
							eval_metric = "logloss",
							verbose = True)

# fit一下得到有正确参数的model, 用这个model predict
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: % 2f%%" % (accuracy * 100.0))

'''
from xgboost import XGBClassifier
from xgboost import plot_importance:  plot_importance(model)

都是这个套路;
	有X，Y两个dataset
	train_test_split()
	用model去fit这些data
	model.predict用在test数据上
	accuracy_score 得到 accuracy

'''
