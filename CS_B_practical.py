import pandas as pd
import matplotlib.pyplot as plt

df =pd.DataFrame(
    {

        "Name": [
            "Braund, Mr. Owen Harris",
            "Allen, Mr. William Henry",
            "Bonnell, Miss. Elizabeth",
        ],
        "Age": [22, 35, 58],
        "Sex": ["male", "male", "female"],
        "Marks": [33,53,46],
    }
)

#print(df["Age"].max())
#print(df.describe())

titanic = pd.read_csv("titanic.csv")
#print(titanic[["Age","Sex"]].groupby("Sex").mean())
#print(titanic.groupby(["Sex","Pclass"])["Age"].count())
#print(titanic["Pclass"].value_counts())




titanic["New_age"]= titanic["Age"]*1.8

#print(titanic[["Age","New_age"]])


air = pd.read_csv("air.csv", parse_dates=True)
air_quality =pd.read_csv("air_quality_long.csv", parse_dates=True)

air_quality["New_date"] = pd.to_datetime(air_quality["date.utc"])


#print(air.shape)
#print(air_quality.dtypes)
air_overall =  pd.concat([air,air_quality])
#print(air_overall.shape)

#print(air_quality["date.utc"].sort_values().head(50))

#print(air["datetime"])
#print(air_quality["New_date"].dt.weekday.head(60))

#print(air_quality["New_date"].max() - air_quality["New_date"].min())

#print(air_quality.dtypes)
'''
print(air_quality.groupby(
    [air_quality["New_date"].dt.weekday, "country"])
    ["value"].mean().plot(style="-o", figsize=(10,5), xlabel="Hello"))
plt.show()




air["New"] = air["station_london"]/air["station_paris"]



air2= air.rename(
    columns={
        "station_antwerp": "BETR801",
        "station_paris": "FR04014",
        "station_london": "London Westminster",
    }
)
summary = air.agg(
    {
        "station_london": ["min", "max", "median", "skew"],
        "station_paris": ["min", "max", "median", "mean"],
    }
)
'''
#print(summary)

#print(titanic.dtypes)
#print(titanic.info())
#print(titanic["Age"].max())

#air.plot.scatter(x="station_london", y="station_paris")

#air.plot.box()

#air.plot.area(figsize=(12,4), subplots= True)


#plt.show()

#scikit-learn
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#datasets
X,y = load_iris(return_X_y=True )
A,b = load_breast_cancer(return_X_y=True )
#print(X['feature_names'])
#print(X1['feature_names'])
#creating training and testing splits
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.6)


# classifiers
lr = LogisticRegression()
rf = RandomForestClassifier()
sv = svm.SVC()
nb = GaussianNB()
dt = tree.DecisionTreeClassifier()
knn = KNeighborsClassifier()

# fitting model to training data
#lr.fit(X_train,y_train)

#model evaluation on test or training dataset
acc = accuracy_score(lr.predict(X_test), y_test)  #testing accuracy
#acc1= lr.score(X_train,y_train) #training accuracy

#printing accuracies
#print(acc*100)
#print(acc1*100)

nb_cv= cross_val_score(sv,A,b,cv=10)
print(round(nb_cv.mean()*100,2))
print(nb_cv.std()*100)












