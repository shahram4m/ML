import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from statsmodels.graphics.regressionplots import influence_plot
# import statsmodels.formula.api as smf
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
import pylab as pl
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from sklearn import preprocessing


df = pd.read_csv("teleCust1000t.csv")
df.head()

print(df['custcat'].value_counts())

# sklearn work with numpy
# create matrix : array of Xs
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
print(X[0:5])

# for one array in Xs have on y here
y = df['custcat'].values

# nomaliz data: simple and important in data

from sklearn.model_selection import train_test_split
# train_test_split : div data per 0.2 and 0.8  and param random_state is : set seed factor for the shuffling applied to the data before applying the split
# If you don't set a seed, it is different each time.
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
k = 4
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print(neigh)

#is predict
yhat = neigh.predict(X_test)
print(yhat[0:5])
print(y[0:5])

# Accuracy evaluation
from sklearn import metrics
                                                            # neigh.predict(X_train) is y' trian
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# if key = 6

# calc k for all n : find out which k is better
Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
for n in range(1,Ks):
    #train model and predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat == y_test)/np.sqrt(yhat.shape[0])
mean_acc
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
