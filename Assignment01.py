#Assignment 1: Customer Subscription Prediction with Machine Learning


import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 2)

import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import matplotlib.pyplot as plt

import time
import warnings
warnings.filterwarnings('ignore')

dt=pd.read_csv("bank-full.csv", delimiter=';')


print(dt.describe(include="all"))

print("All_training: \n",dt.columns)

print("\n\nSample data : \n")
print(dt.sample(5))

print(dt.dtypes)
#We will convert some of the object type data into numerical data as the columns contain numbers

print(pd.isnull(dt).sum())

#Missing values in contact, poutcome, pdays as given in variables table
print(dt['contact'].value_counts())
#13020 unknown values
print(dt['poutcome'].value_counts())
print(dt['pdays'].value_counts())
print(dt['y'].value_counts())

#y
def discrete(i):
    if i['y'] == 'no':
        return 0
    if i['y'] == "yes":
        return 1
dt['y']=dt.apply(discrete,axis=1)


#print(dt[dt['contact'] == 'cellular']['y'].value_counts())
#0    24916
#1     4369
#print("Percentage of term deposit taken where contact is cellular out of all cellular contacts:", dt["y"][dt["contact"] == 'cellular'].value_counts(normalize = True)[1]*100)
#4369/(4369+24916)



names = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'y']

correlations = dt.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,8,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


print(dt['marital'].value_counts())


def cl_contact(i):
    if i['contact'] == 'telephone':
        return 0
    if i['contact'] == "cellular":
        return 1
dt['contact']=dt.apply(cl_contact,axis=1)



def cl_marital(i):
    if i['marital'] == 'divorced':
        return 0
    if i['marital'] == "single":
        return 1
    if i['marital'] == 'married':
        return 2
dt['marital']=dt.apply(cl_marital,axis=1)

def cl_pout(i):
    if i['poutcome'] == 'failure':
        return 0
    if i['poutcome'] == "success":
        return 1
    if i['poutcome'] == 'other':
        return 2
dt['poutcome']=dt.apply(cl_pout,axis=1)


def cl_default(i):
    if i['default'] == 'no':
        return 0
    if i['default'] == "yes":
        return 1
dt['default']=dt.apply(cl_default,axis=1)


def cl_housing(i):
    if i['housing'] == 'no':
        return 0
    if i['housing'] == "yes":
        return 1
dt['housing']=dt.apply(cl_housing,axis=1)


def cl_loan(i):
    if i['loan'] == 'no':
        return 0
    if i['loan'] == "yes":
        return 1
dt['loan']=dt.apply(cl_loan,axis=1)

print(dt.sample(10))

sbn.countplot(x='age', data=dt)
plt.show()




#print("Percentage of term deposit taken where contact is cellular out of all cellular contacts:", dt["y"][dt["contact"] == 'cellular'].value_counts(normalize = True)[1]*100)
#4369/(4369+24916)

sbn.barplot(x="age", y='y', data=dt)
plt.tight_layout()
plt.show()

print(dt[dt['age'] >=20]['y'].value_counts())
print(dt[dt['age'] >=30]['y'].value_counts())
print(dt[dt['age'] >=40]['y'].value_counts())

print(dt[dt['age'] >=60]['y'].value_counts())
print(dt[dt['age'] >=70]['y'].value_counts())
print(dt[dt['age'] >=80]['y'].value_counts())
print(dt[dt['age'] >=90]['y'].value_counts())

print('\n\n\n')
dt=dt[dt['age']>=25]


print(dt['duration'].value_counts())

print(dt[dt['duration'] <=10]['y'].value_counts())
print(dt[dt['duration'] <=20]['y'].value_counts())
print(dt[dt['duration'] <=30]['y'].value_counts())
print(dt[dt['duration'] <=40]['y'].value_counts())
print(dt[dt['duration'] <=50]['y'].value_counts())
print(dt[dt['duration'] <=60]['y'].value_counts())


dt=dt[dt['duration']>60]

print('\n\n')


print(dt[dt['pdays'] == -1]['y'].value_counts())

print(dt['month'].unique())
month_mapping={'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12}

data=[dt]
for dataset in data:
    dataset['month'] = dataset['month'].map(month_mapping)
print("\n\n")


sbn.barplot(x="poutcome", y='y', data=dt)
plt.tight_layout()
plt.show()


print("Percentage of term deposit taken where outcome is other out of all other outcomes:", dt["y"][dt["poutcome"] == 2].value_counts(normalize = True)[1]*100)
print("Percentage of term deposit taken where outcome is success out of all other outcomes:", dt["y"][dt["poutcome"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of term deposit taken where outcome is failure out of all other outcomes:", dt["y"][dt["poutcome"] == 0].value_counts(normalize = True)[1]*100)
print('\n\n')

dt1 = dt.drop(dt[dt['poutcome'] == 2].index, axis = 0, inplace = False)


print(dt1.describe())
print('\n\n')
print("Jobs :",dt1['education'].unique())
print('\n\n')

dt1["education"].replace('unknown',np.nan,inplace=True)
dt2=dt1.dropna(subset=["education"])

education_mapping = {"primary":1,"secondary":2,"tertiary":3}
data1=[dt2]
for dataset in data1:
    dataset['education'] = dataset['education'].map(education_mapping)

print(dt2.sample(10))



sbn.barplot(x="education", y='y', data=dt2)
plt.tight_layout()
plt.show()

print('\n\n')
print(dt2['job'].unique())
print('\n\n')

dt2["job"].replace('unknown',np.nan,inplace=True)

dt3=dt2.dropna(subset=["job"])

sbn.countplot(x='job', data=dt2)
plt.show()

sbn.barplot(x="job", y='y', data=dt2)
plt.tight_layout()
plt.show()


#print("\n\nRelation between pdays and poutcome")
dt3["poutcome"]  = dt3["poutcome"].fillna(-0.5)


sbn.barplot(x="poutcome", y='y', data=dt3)
plt.tight_layout()
plt.show()
#the poutcome dependency on NaN values is almost similar to the failure outcome
#dropping all unknown values
#dt3=dt3[dt3['poutcome']!=-0.5]

#we can also replace all unknown values by 0, as both have same trend
dt3["poutcome"].replace(-0.5,0,inplace=True)


sbn.barplot(x="month", y="y", data=dt3)
plt.tight_layout()
plt.show()
#3,9,10,12 have more percentage of term deposit subscribed

def cl_month(i):
    if i['month'] == 3 or i['month'] == 9 or i['month'] == 10 or i['month'] == 12:
        return 1
    else:
        return 0
dt3['month']=dt3.apply(cl_month,axis=1)

sbn.barplot(x="contact", y="y", data=dt3)
plt.tight_layout()
plt.show()

#does not give any specific information
dt4=dt3.drop(['contact'],axis=1)

print(dt4.sample(11))
print(dt4.info())
dt4 = pd.get_dummies(dt4,columns=['job'],drop_first=True)


print('\n\n Training the model...')

from sklearn.model_selection import train_test_split

input_predictors = dt4.drop(['y'], axis=1)
output_target = dt4["y"]


x_train, x_val, y_train, y_val = train_test_split(input_predictors, output_target,
                                                  test_size=0.25, random_state=7)



#Testing Different Models

#1) Logistic Regression
#2) Gaussian Naive Bayes
#3) Support Vector Machines
#4) Decision Tree Classifier
#5) Random Forest Classifier
#6) KNN or k-Nearest Neighbors
#7) LDA or Linear Discriminant Analysis

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#MODEL-1) LogisticRegression
#------------------------------------------
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print( "MODEL-1: Accuracy of LogisticRegression : ", acc_logreg)

print("\n\n")
print("confusion_matrix = \n")
print( confusion_matrix(y_val, y_pred))

scaler = StandardScaler().fit(x_train)
X_train_scaled = scaler.transform(x_train)
model = SVC()

model.fit(X_train_scaled, y_train)   #Training of algorithm

x_val_scaled=scaler.transform(x_val)
predictions = model.predict(x_val_scaled)
print("All predictions done successfully by SVM Machine Learning Algorithms")
print("\n\nAccuracy score %f" % accuracy_score(y_val, predictions))

print("\n\n")
print(" scaled confusion_matrix = \n")
print( confusion_matrix(y_val, predictions))

#Using the components of the confusion matrix, we can define the various metrics used for evaluating classifiers—accuracy, precision, recall, and F1 score.
precision=7828/(174+7828)
recall=7828/(7828+305)
#The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
F1 = 2 * (precision * recall) / (precision + recall)
print('\n\n')
print("F1 score", F1)