import pandas as pd
import matplotlib.pyplot as plt
'''
data = pd.DataFrame(
    {
        "Age": [22, 35, 58],
        "Sex": ["male", "male", "female"],
        "Marks": [33, 53, 46],
        "Height": [6.2,4.5,5.5],
    }
)

#print(data["Age"].describe())
print(data.head(20))

titanic = pd.read_csv("titanic.csv")
air = pd.read_csv("air.csv", parse_dates=True)
air_long =pd.read_csv("air_quality_long.csv", parse_dates=True)

air["New_date"] = pd.to_datetime(air["datetime"])

air_long["New"] = pd.to_datetime(air_long["date.utc"])
print(air_long.dtypes)
print(air_long.groupby(
    [air_long["New"].dt.weekday, "country"])["value"].mean().plot(style="-o", figsize=(12,4), xlabel="hello"))

plt.show()
#print(air["New_date"].dt.weekday)
#print(air.sort_values("station_london").head(50))
print(air["New_date"].max() - air["New_date"].min())
#print(air_long["date.utc"])

titanic["New_age"]= titanic["Age"] *1.5

air["New"] = air["station_paris"]/air["station_london"]
filtered = titanic[titanic["Age"]>30]
#print(filtered.plot())
#plt.show()

summary = titanic.agg(
    {
        "Age": ["min", "max", "median", "skew"],
        "New_age": ["min", "max", "median", "mean"],
    }
)

#print(summary)
#titanic.plot()
#air.plot.scatter(x="station_london", y="station_paris")

#air.plot.area(figsize=(12,4), subplots=True)

#plt.show()

#print(titanic)

air_overall = pd.concat([air_long,air], axis=0)
#print(air_overall)


air_quality = pd.read_csv("air_quality_long.csv", parse_dates=True)
air_quality = air_quality.rename(columns={"date.utc": "datetime"})
air_quality["datetime"] = pd.to_datetime(air_quality["datetime"])
#print(air)
print(air_quality["datetime"].max() - air_quality["datetime"].min())
air_quality["months"] = air_quality["datetime"].dt.month

print(air_quality.groupby(
    [air_quality["datetime"].dt.weekday, "location"])["value"].mean())
#print(air_quality)
#fig, axs = plt.subplots(figsize=(12, 4))
air_quality.groupby(air_quality["datetime"].dt.hour)["value"].mean().plot(
    kind='area', rot=0, figsize=(12,4), xlabel= "Hello", ylabel="hi"
)
plt.show()
#print(air.pivot_table(values="value", index="location", columns="parameter", aggfunc="mean"))

air_quality_overall = pd.concat([air, air_quality], axis=1)
#print(air_quality_overall)
#print(air_quality_overall.sort_values("date.utc"))


import numpy as np
from numpy import pi

a = np.array([[(2,3,4,6),(1.5,4,5.5,6),(5,3,1,6),(5,3,1,6)],[(5,3,1,6),(2,3,4,5),(1.5,4,5.5,5),(5,3,1,5)]])
b = np.array((2,3,2))
#print(a)
#print(np.zeros((3,4)))
#print(np.ones((3,4)))
#print(np.empty((3,4)))
#print(np.arange(10).reshape(2,5))
#print(np.arange(10,30,2).reshape(2,5))
#print(np.arange(0,2,0.03))
#print(np.linspace(0, 2, 100))
#print(np.linspace(0, 2 * pi, 10))
'''

###  Numpy   #####

import numpy as np
from numpy import pi
my = np.array([[(3,7,5),(2,9,1)],[(4,5,9),(2,9,1)]])
a = np.array([1, 2, 3, 4,5,6,7])
b = np.array([5, 6, 7, 8])

#print(np.sort(my))
#print(np.concatenate([a,b]))

#print(a[0:3])
#print(a[3:])
#print(a[-3:])

test = np.arange(15).reshape(3,5)
#print(test)
#print(test[0:3, 0:2])
#print((test<10) | (test>12))

test2 = np.nonzero(my)
#print(test2)

#li = list(zip(test2[0],test2[1]))
#for i in li:
#    print(i)

#print(my.shape)
#print(my.ndim)
#print(my.dtype)

#print(np.zeros((3,4)))
#print(np.ones((3,4)))
#print(np.empty((3,4)))

#print(np.arange(10,35,5))
#print(np.arange(0,2, 0.3))
#print(np.linspace(0,2*pi,20))


from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris( )
X = iris['data']
y=  iris['target']

X1,y1 = load_breast_cancer(return_X_y=True)

X_train,X_test, y_train, y_test = train_test_split(X,y, train_size=0.8)

sv = svm.SVC()
nb = GaussianNB()
dt = tree.DecisionTreeClassifier()
knn = KNeighborsClassifier()
lr = LogisticRegression()
rf = RandomForestClassifier()


nb.fit(X_train, y_train)

nb_train_acc = accuracy_score(nb.predict(X_train),y_train)
nb_test_acc = accuracy_score(nb.predict(X_test),y_test)

#print(nb_train_acc)
#print(nb_test_acc)

#K-fold cross validation
dt_cv = cross_val_score(dt, X1, y1, cv=10)
print(round(dt_cv.mean()*100, 2))
print(dt_cv.std()*100)














