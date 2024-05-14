import pandas as pd
import matplotlib.pyplot as plt

my_class = pd.DataFrame(
    {
       "Name" : ["ALi", "Faiza", "Asad"],
        "Age" : [23,21,34],
        "Marks" : [34, 54,21],
    }
)
#print(my_class[["Age","Marks"]])
#print(my_class.shape)
#print(my_class.info())
#print(my_class.dtypes)
#print(my_class.describe())

#print(my_class["Age"])

titanic = pd.read_csv("titanic.csv")
air = pd.read_csv("air.csv", parse_dates=True)
air_quality = pd.read_csv("air_quality_long.csv", parse_dates=True)

air_quality["New_date"] = pd.to_datetime(air_quality["date.utc"])
#print(air_quality.sort_values("date.utc").head(50))
#print(air["datetime"].max())
#print(air_quality["New_date"].max() - air_quality["New_date"].min())
#print(air_quality["New_date"].dt.month)



#print(air_quality.groupby(["location",air_quality["New_date"].dt.weekday])["value"].mean().plot(kind="bar", figsize=(12,4), xlabel="Hello", ylabel="Hello"))

plt.show()
air_overall = pd.concat([air_quality,air])

#print(air_overall.shape)






air["New_london"] = air["station_london"]*1.5

summary = air.agg(
    {
        "station_london": ["min", "max", "median", "skew"],
        "station_paris": ["min", "max", "median", "mean"],
    }
)

#print(summary)

air2= air.rename(
    columns={
        "station_antwerp": "BETR801",
        "station_paris": "FR04014",
        "station_london": "London Westminster",
    }
)
#print(air2)
#print(air.plot())
#print(air["station_paris"].plot())
#print(air.plot.scatter(x="station_paris", y="station_london"))
#print(air.plot.area(figsize=(12,4), subplots= True))
#plt.show()

#print(titanic.head(20))
#print(titanic.info())
#print(titanic.describe())
#print(titanic[titanic["Age"]>50])




import numpy as np
from numpy import pi
my = np.array([[(3,7,5),(2,9,1)],[(4,5,9),(2,9,1)]])
a = np.array([1, 2, 3, 4,5,6,7])
b = np.array([5, 6, 7, 8])

#print(a[0:3])
#print(a[3:])
#print(a[-3:])

test = np.arange(15).reshape(3,5)
#print(test[0:3, 0:4])
#print(test[(test<10) | (test>12)])

test2 = np.nonzero((test>2) & (test<10))
#print(test2)

coord = list(zip(test2[0],test2[1]))

#for i in coord:
  #  print(i)

#print(np.sort(my))
#print(np.concatenate([a,b]))

#print(my.shape)
#print(my.ndim)
#print(my.dtype)

#print(np.zeros((3,4)))
#print(np.ones((3,4)))
#print(np.empty((3,4)))

#print(np.arange(10,35,5))
#print(np.arange(0,2, 0.3))
#print(np.linspace(0,2*pi,20))


#UCI machine learning repository

from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score # K-fold cross validation
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score

iris = load_iris( )
#print(iris['target'])
X= iris['data']
y = iris['target']
X1,y1 = load_breast_cancer(return_X_y=True)

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, train_size=0.6)

nb = GaussianNB()
lr = LogisticRegression()
rf = RandomForestClassifier()
knn = KNeighborsClassifier()
svm = svm.SVC()
dt = tree.DecisionTreeClassifier()

nb.fit(Xtrain,ytrain)

nb_acc = nb.score(Xtrain,ytrain)
acc = accuracy_score(nb.predict(Xtest), ytest)

#print(nb_acc*100)
#print(acc*100)

dt_cv= cross_val_score(dt, X1,y1, cv=5)
print(round(dt_cv.mean()*100,2))
print(dt_cv.std()*100)