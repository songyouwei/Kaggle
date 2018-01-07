import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

predictors = train.drop(['label'], axis=1)
target = train["label"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)

knn = KNeighborsClassifier()
algo = knn
algo.fit(x_train, y_train)

ids = list(range(1,len(test)+1))
predictions = algo.predict(test)

output = pd.DataFrame({ "ImageId": ids, 'Label': predictions })
output.to_csv('submission.csv', index=False)
