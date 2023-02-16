import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,
    classification_report, plot_confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('oasis_longitudinal.csv')
# Data Cleaning
todel = ['Hand']
dataset = data.drop(todel, axis=1)
#   Pre Processing
# Missing value treatment
# checking missing values in each column
print(dataset.isna().sum())
# for better understanding lets check the percentage of missing values in each column
print(round(dataset.isnull().sum() / len(dataset.index), 2) * 100)


# So, we have to impute missing values in SES and MMSE. Let's analyze SES column
# Plotting distribution of SES
def univariate_mul(var):
    fig = plt.figure(figsize=(16, 12))
    cmap = plt.cm.Blues
    cmap1 = plt.cm.coolwarm_r
    ax1 = fig.add_subplot(221)
    fig.add_subplot(212)
    dataset[var].plot(kind='hist', ax=ax1, grid=True)
    ax1.set_title('Histogram of ' + var, fontsize=14)

    ax2 = sns.distplot(dataset[[var]], hist=False)
    ax2.set_title('Distribution of ' + var)
    plt.show()


# let's see the distribution of SES to decide which value we can impute in place of missing values.

univariate_mul('SES')
dataset['SES'].describe()
# As SES has values of integer type, so we cannot impute float value of mean, but we can impute median in place as both
# median and mean have very close values and median in this case is the most representative value of SES.

# imputing missing value in SES with median
dataset['SES'].fillna((dataset['SES'].median()), inplace=True)
# Next we will analyze another column having missing values i.e., MMSE
univariate_mul('MMSE')
dataset['MMSE'].describe()
# MMSE also has integer values, so we cannot impute float. So we will impute it with median value
# imputing MMSE with median values
dataset['MMSE'].fillna((dataset['MMSE'].median()), inplace=True)
# Now, lets check the percentage of missing values in each column
round(dataset.isnull().sum() / len(dataset.index), 2) * 100
# correlation
plt.figure(figsize=(14, 8))
sns.heatmap(dataset.corr(), annot=True)
plt.show()

"""#Outlier Deletion"""

dataset = dataset.drop(labels=[0, 1, 2, 3], axis=0)

dataset['SES'] = dataset['SES'].astype('object')
dataset['CDR'] = dataset['CDR'].astype('object')

for column in dataset.columns:
    if dataset[column].dtype == type(object):
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])

predictors = dataset.drop(["eTIV", "ASF"], axis=1)
target = dataset["Group"]
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.3, random_state=0)

"""## Random Forest"""

rf = RandomForestClassifier(random_state=1000)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)
score_rf = round(accuracy_score(Y_pred_rf, Y_test) * 100, 2)
print("The accuracy score achieved using random forest classifier is: " + str(score_rf) + " %")
print(confusion_matrix(Y_test, Y_pred_rf))
print(classification_report(Y_test, Y_pred_rf))
plot_confusion_matrix(rf, X_test, Y_test)
plt.show()

sf = svm.SVC(random_state=0).fit(X_train, Y_train)
sf.fit(X_train, Y_train)
Y_pred_sf = sf.predict(X_test)
score_sf = round(accuracy_score(Y_pred_sf, Y_test) * 100, 2)

print("The accuracy score achieved using Support Vector Machine is: " + str(score_sf) + " %")
print(confusion_matrix(Y_pred_sf, Y_test))
print(classification_report(Y_pred_sf, Y_test))

classifier = GaussianNB()
classifier.fit(X_train, Y_train)
pred_NB = classifier.predict(X_test)
acc_NB = round(accuracy_score(pred_NB, Y_test) * 100, 2)
print("The accuracy score achieved using Naive_Bayes is: " + str(acc_NB) + " %")
print(confusion_matrix(pred_NB, Y_test))
print(classification_report(pred_NB, Y_test))

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, Y_train)
pred_KNN = knn.predict(X_test)
acc_KNN = round(accuracy_score(pred_KNN, Y_test) * 100, 2)
print("The accuracy score achieved using KNeighborsClassifier is: " + str(acc_KNN) + " %")
print(confusion_matrix(pred_KNN, Y_test))
print(classification_report(pred_KNN, Y_test))

modelLogistic = LogisticRegression()
modelLogistic.fit(X_train, Y_train)
pred_LOG = modelLogistic.predict(X_test)
acc_LOG = round(accuracy_score(pred_LOG, Y_test) * 100, 2)
print("The accuracy score achieved using Logistic Regression is: " + str(acc_LOG) + " %")
print(confusion_matrix(pred_LOG, Y_test))
print(classification_report(pred_LOG, Y_test))

FOLDS = 5

parametros_xgb = {
    "learning_rate": [0.01, 0.025, 0.005, 0.5, 0.075, 0.1, 0.15, 0.2, 0.3, 0.8, 0.9],
    "max_depth": [3, 5, 8, 10, 15, 20, 25, 30, 40, 50],
    "n_estimators": range(1, 1000)
}

model_xgb = XGBClassifier(eval_metric='mlogloss')

xgb_random = RandomizedSearchCV(estimator=model_xgb, param_distributions=parametros_xgb, n_iter=100, cv=FOLDS,
                                verbose=0, random_state=42, n_jobs=-1, scoring='accuracy')
xgb_random.fit(X_train, Y_train)

model_xgb = xgb_random.best_estimator_
model_xgb.fit(X_train, Y_train)
pred_xgb = model_xgb.predict(X_test)
acc_xgb = round(model_xgb.score(X_test, Y_test) * 100, 2)
print("The accuracy score achieved using XGBoost is: " + str(acc_xgb) + " %")
print(confusion_matrix(pred_xgb, Y_test))
print(classification_report(pred_xgb, Y_test))

plt.title("Model Selection")

# Data
subject = ['RF', 'SVM', 'NB', 'KNN', 'LR', 'XG']
marks = [91, 50, 89, 55, 87, 88]

# Plot
plt.bar(subject, marks, width=0.50, edgecolor='k', linewidth=2)

plt.xlabel("Model")
plt.ylabel("Accuracy")

# Create the graph ticks with a list comprehension
plt.yticks(ticks=[x * 10 for x in range(11)])

# Render
plt.savefig("plot.png")
plt.show()