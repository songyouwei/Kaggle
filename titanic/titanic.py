import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train = train.drop(['Cabin','Ticket','Name','Fare'], axis = 1)
test = test.drop(['Cabin','Ticket','Name','Fare'], axis = 1)

def simplify_ages(df):
    bins = (0, 5, 12, 18, 24, 35, 60, 120)
    group_names = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

simplify_ages(train)
simplify_ages(test)

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['Age'] = train['Age'].map(age_mapping)
test['Age'] = test['Age'].map(age_mapping)

sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.fillna({'Embarked':1, 'Age':round(train['Age'].mean(),1)}, inplace=True)
test.fillna({'Embarked':1, 'Age':round(test['Age'].mean(),1)}, inplace=True)


predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)

def choose_algo(algos):
	algo_dict = {}
	for algo in algos:
		algo.fit(x_train, y_train)
		y_pred = algo.predict(x_val)
		acc_algo = round(accuracy_score(y_pred, y_val) * 100, 2)
		algo_dict[acc_algo] = algo
		print('algo: {0} , acc: {1}'.format(algo.__class__.__name__, acc_algo))
	maxacc = sorted(algo_dict.keys(), reverse=True)[0]
	return algo_dict[maxacc]

gaussian = GaussianNB()
logreg = LogisticRegression()
svc = SVC()
linear_svc = LinearSVC()
decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()
knn = KNeighborsClassifier()
sgd = SGDClassifier()
gbk = GradientBoostingClassifier()

algo = choose_algo([logreg, svc, decisiontree, randomforest, knn, gbk])
print(algo) # GradientBoostingClassifier

#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = algo.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)
