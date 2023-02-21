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



df= pd.read_csv("50_Startups.csv")
# print(df.head())
# print(df.corr())

# R&D Spend,Administration,Marketing Spend,State,Profit
cdf = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit']]
cdf = cdf.rename(columns={"R&D Spend": "Spend", "Marketing Spend": "Marketing_Spend"})

def show():
    sns.set_style(style="whitegrid")
    sns.pairplot(df)
    plt.show()

    plt.scatter(cdf.Spend, cdf.Profit, color='blue')
    plt.xlabel('Spend')
    plt.ylabel('Profit')
    plt.show()

    train, trainx, trainy,  test, regr = None
def createMask():
    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]
    # print(msk)
    # print([~msk])
    # print(train)
    # print(test)

    fig = plt.figure()
    pho1 = fig.add_subplot(111)
    pho1.scatter(train.Spend, train.Profit, color='blue')
    pho1.scatter(test.Spend, test.Profit, color='red')
    plt.xlabel('Spend')
    plt.ylabel('Profit')
    plt.show()


def LinearRegression():
    regr = linear_model.LinearRegression()
    trainx = np.asanyarray(train[['Spend']])
    trainy = np.asanyarray(train[['Profit']])
    regr.fit(trainx, trainy)
    print('Coefficients teta1', regr.coef_)
    print('Intercept', regr.intercept_)

    # calc line
    plt.scatter(train.Spend, train.Profit, color='blue')
    plt.plot(trainx, regr.coef_[0][0] * trainx + regr.intercept_[0], '-r')
    #                      y = θ1     *  X     +  θ0
    plt.xlabel('Spend')
    plt.ylabel('Profit')
    plt.show()

#---------------- Eval
def Evaloation():
    testx = np.asanyarray(test[['Spend']])
    testy = np.asanyarray(test[['Profit']])
    testy_ = regr.predict(testx)

    print('Main absolut error : %.2f' % np.mean(np.absolute(testy_ - testy)))
    print('Residual sum of squares (MSE) : %.2f' % np.mean((testy_ - testy) ** 2 ))
    print(' R2 result : %.2f ' % r2_score(testy , testy_ ) )

#--------------------------multi line
def createMultiLine():
    regr = linear_model.LinearRegression()
    trainx = np.asanyarray(train[['Spend', 'Administration', 'Marketing_Spend']])
    trainy = np.asanyarray(train[['Profit']])
    regr.fit(trainx, trainy)
    print('Coefficients θ1', regr.coef_)
    print('Intercept', regr.intercept_)

    # calc line
    plt.scatter(train.Spend, train.Profit, color='blue')
    plt.plot(trainx, regr.coef_[0][0] * trainx + regr.intercept_[0], '-r')
    #                      y = θ1     *  X     +  θ0
    plt.xlabel('Spend')
    plt.ylabel('Profit')
    plt.show()

#---------------- Eval multiline
# ------------------------------predict test
def predict_test():
    y_hat = regr.predict(test[['Spend', 'Administration', 'Marketing_Spend']])
    x = np.asanyarray(test[['Spend', 'Administration', 'Marketing_Spend']])
    y = np.asanyarray(test[['Profit']])
    print("Residual sum of squares : %.2f" % np.mean((y_hat - y) ** 2))
    print('Variance score : %.2f' % regr.score(x, y))

createMask()
LinearRegression()
createMultiLine()
predict_test()
