import numpy as np
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import normalize


def parse():
    data_file = open('../data/data.csv', 'rb')
    first_line = True
    target_vals = []
    data = []
    for line in data_file:
        if first_line:
            first_line = False
            continue
        else:
            target_vals.append(line[0])
            data.append(line[2:].split(","))

    return data, target_vals


def all_model(alpha):
    SGD = linear_model.SGDClassifier(alpha=alpha)
    preceptron = linear_model.SGDClassifier(
        loss='perceptron', penalty='elasticnet', alpha=alpha)
    ensemble = GradientBoostingRegressor()
    RF = RandomForestClassifier(n_estimators=10)
    gnb = GaussianNB()
    ext = ExtraTreesClassifier(n_estimators=10, max_depth=None,
                               min_samples_split=1, random_state=0)
    # ,(ensemble,'ensemble')]
    # 
    return [(SGD, 'StocasticG'), (preceptron, 'preceptron'), (gnb, 'GaussianNB'), (RF, 'RandomForest'),(ext,"ExtraTrees")]


def trails(data, target_vals, bs, alpha):
    list_accuracy = []
    for train_index, test_index in bs:

	    train = data[train_index]
	    test = data[test_index]
	    train_target_vals = target_vals[train_index]
	    test_traget_vals = target_vals[test_index]
	    models = all_model(alpha)
	    for model, model_type in models:
	        model.fit(train, train_target_vals)
	        predict = model.predict(test)
	        accuracy = metrics.accuracy_score(test_traget_vals, predict)
	        list_accuracy.append(accuracy)
	        # print model_type, " = ", accuracy
    return max(list_accuracy)


if __name__ == "__main__":
    data, targets = parse()
    data = np.array(data)
    data = data.astype(np.float)
    data = normalize(data,axis=1)
    #print (data[1][0])
    targets = np.array(targets)
    #print (targets[0])
    alpha_vals = np.linspace(750, 1000, 20)
    bs = cv.Bootstrap(targets.size, n_iter=100)
    max_vals = []
    for alpha in alpha_vals:
        print '\n', alpha
        max_vals.append(trails(data, targets, bs, alpha))
	print max(max_vals)

