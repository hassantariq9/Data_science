import pandas as pd
import matplotlib.pyplot as plt

my = pd.DataFrame(
{

        "Name": [
            "Braund, Mr. Owen Harris",
            "Allen, Mr. William Henry",
            "Bonnell, Miss. Elizabeth",
        ],
        "Age": [22, 35, 58],
        "Sex": ["male", "male", "female"],
        "Marks": [33,53,46],
        "Height": [4.5,6.1,5.5],
    }
)

#print(my["Marks"].describe())
#print(my["Height"].max())

titanic = pd.read_csv("titanic.csv")

#print(titanic["Name"].str.split(" ").str.get(3))
#print(titanic[titanic["Name"].str.contains("Countess")])

#print(titanic.groupby("Sex").mean())


air = pd.read_csv("air.csv", parse_dates=True)
air_quality = pd.read_csv("air_quality_long.csv", parse_dates=True)

air_quality["New_date"] = pd.to_datetime(air_quality["date.utc"])

air["New"] = air["station_paris"]*1.5
#print(air["datetime"])

#print(air_quality["New_date"].max() - air_quality["New_date"].min())
#print(air_quality["New_date"].dt.weekday)

'''
print(air_quality.groupby(
    [air_quality["New_date"].dt.weekday , "location"])
      ["value"].mean().plot(kind="bar", figsize=(12,5), ylabel="hello"))
'''
plt.show()
#print(air_quality["date.utc"].max() - air_quality["date.utc"].min())

air_overall = pd.concat([air,air_quality])

#print(air_overall.shape)

air2= air.rename(
        columns= {"station_paris" : "paris", "station_london" : "london"}
)

summary= air.agg(
    {
    "station_london" : ["mean","skew","max"],
    "station_paris" : ["min","median","mean"],
    }

)

#print(air2)
#air.plot.scatter(x="station_paris", y="station_london")
#air.boxplot()
#titanic.plot.area()
#air.plot.area(figsize=(12,4), subplots= True)
#titanic.plot()

#plt.show()
#print(titanic[["Age","Sex"]].shape)
#print(titanic.describe())
#print(titanic.dtypes)
#air["New_london"]= air["station_london"]*1.5
#print(air[["station_london","station_paris"]].median())


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
#print(test[test<10])
#print(test[(test<10) | (test>12)])

#print(test)
test2= np.nonzero(test<12)

#print(test2)

co = list(zip(test2[0], test2[1]))
#for m in co:
    #print(m)

#oord = list(zip(test2[0],test2[1]))
#for i in coord:
#    print(i)

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

#UCI machine lerning repository
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score

#datasets
iris = load_iris( )
X = iris['data']
y= iris['target']
A,b= load_breast_cancer(return_X_y=True)
#splitting the datasets
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, train_size=0.7)
Atrain, Atest, btrain, btest = train_test_split(A,b, train_size=0.6)
#intialize the classifiers
lr = LogisticRegression()
rf = RandomForestClassifier()
knn = KNeighborsClassifier()
nb = GaussianNB()
dt = tree.DecisionTreeClassifier()
sv = svm.SVC()

#fitting model
lr.fit(Xtrain,ytrain)
rf.fit(Atrain,btrain)

#evaluating the model
rf_acc= rf.score(Atest,btest)
acc = lr.score(Xtrain,ytrain)
lr_acc = accuracy_score(lr.predict(Xtest),ytest)
'''
print("************************************")
print("****** Random forest ***************")
print(rf_acc*100)
print("************************************")
print("****** logistic regression *********")
print(acc*100)
print(lr_acc*100)
print("************************************")
print("************************************")
'''

dt_cv = cross_val_score(dt, A, b, cv=10)
print(dt_cv)
print(round(dt_cv.mean()*100,2))
print(dt_cv.std()*100)










