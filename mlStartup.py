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
regr = linear_model.LinearRegression()


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

    return train, test
    # print(msk)
    # print([~msk])
    # print(train)
    # print(test)

    # fig = plt.figure()
    # pho1 = fig.add_subplot(111)
    # pho1.scatter(train.Spend, train.Profit, color='blue')
    # pho1.scatter(test.Spend, test.Profit, color='red')
    # plt.xlabel('Spend')
    # plt.ylabel('Profit')
    # plt.show()


def LinearRegression():
    train, test = createMask()
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
    train, test = createMask()
    testx = np.asanyarray(test[['Spend']])
    testy = np.asanyarray(test[['Profit']])
    testy_ = regr.predict(testx)

    print('Main absolut error : %.2f' % np.mean(np.absolute(testy_ - testy)))
    print('Residual sum of squares (MSE) : %.2f' % np.mean((testy_ - testy) ** 2 ))
    print(' R2 result : %.2f ' % r2_score(testy , testy_ ) )

#--------------------------multi line
def createMultiLine():
    train, test = createMask()
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
    train, test = createMask()
    y_hat = regr.predict(test[['Spend', 'Administration', 'Marketing_Spend']])
    x = np.asanyarray(test[['Spend', 'Administration', 'Marketing_Spend']])
    y = np.asanyarray(test[['Profit']])
    print("Residual sum of squares : %.2f" % np.mean((y_hat - y) ** 2))
    print('Variance score : %.2f' % regr.score(x, y))


def ploynomial():
    clf = linear_model.LinearRegression()
    train, test = createMask()
    trainx = np.asanyarray(train[['Marketing_Spend']])
    trainy = np.asanyarray(train[['Profit']])

    testx = np.asanyarray(test[['Marketing_Spend']])
    testy = np.asanyarray(test[['Profit']])
    # print(trainx[:3])

    poly = PolynomialFeatures(degree=2)
    # create trainx**2
    trian_x_poly = poly.fit_transform(trainx)
    # print(trian_x_poly)

    train_y = clf.fit(trian_x_poly, trainy)
    print('Coefficients:', clf.coef_)
    print('Intercept:', clf.intercept_)

    plt.scatter(train.Marketing_Spend, train.Profit, color='blue')
    xx = np.arange(0.0, 10000.0, 0.1)
    yy = clf.intercept_[0] + clf.coef_[0][1] * xx + clf.coef_[0][2] * np.power(xx, 2)
    plt.plot(xx, yy, '-r')
    plt.xlabel('Marketing_Spend')
    plt.ylabel('Profit')
    plt.show()


# def eval_polynemial():

#------------------------------Regg non linear reg
def non_linear_regg():
    # x = np.arange(-5.0, 5.0, 0.1)
    # y = 2 * (x) + 3
    #
    # # create noise
    # y_noise = 2 * np.random.normal(size=x.size)
    # ydata = y + y_noise
    #
    # plt.plot(x, ydata, 'bo')
    # plt.plot(x, y, 'r')
    # plt.xlabel('dep var')
    # plt.ylabel('indep var')
    # plt.show()
    #
    # #------------------------------Regg non linear reg example log(x)
    # #x**3 + x**2 + x + 3
    # x = np.arange(-5.0, 5.0, 0.1)
    # y = 1 * (x**3) + 1 * (x**2) + 1 * x + 3
    # y_noise = 20 * np.random.normal(size=x.size)
    # ydata = y + y_noise
    #
    # plt.plot(x, ydata, 'bo')
    # plt.plot(x, y, 'r')
    # plt.xlabel('dep var')
    # plt.ylabel('indep var')
    # plt.show()

    #------------------------------Regg non linear reg example
    df = pd.read_csv("china_gdp.csv")
    plt.figure(figsize=(8, 5))
    x_data, y_data = (df['Year'].values, df['Value'].values)
    plt.plot(x_data, y_data, 'ro')
    plt.xlabel('Year')
    plt.ylabel('GDP')
    plt.show()

    #Exponential
    X = np.arange(-5.0, 5.0, 0.1)
    ##You can adjust the slope and intercept to verify the changes in the graph
    Y= np.exp(X)
    plt.plot(X,Y)
    plt.ylabel('Dependent Variable')
    plt.xlabel('Independent Variable')
    plt.show()

    # Logarithmic
    X = np.arange(-5.0, 5.0, 0.1)
    Y = np.log(X)
    plt.plot(X,Y)
    plt.ylabel('Dependent Variable')
    plt.xlabel('Independent Variable')
    plt.show()

    #Sigmoidal/Logistic
    X = np.arange(-5.0, 5.0, 0.1)
    Y = 1-4/(1+np.power(3, X-2))
    plt.plot(X, Y)
    plt.ylabel('Dependent Variable')
    plt.xlabel('Independent Variable')
    plt.show()

def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y

def non_linear_regg_inAct():
    df = pd.read_csv("china_gdp.csv")
    x_data, y_data = (df["Year"].values, df["Value"].values)
    beta_1 = 0.10
    beta_2 = 2000.0
    #logistic function
    Y_pred = sigmoid(x_data, beta_1 , beta_2)
    #plot initial prediction against datapoints
    plt.plot(x_data, Y_pred*15000000000000.)
    plt.plot(x_data, y_data, 'ro')
    # plt.show()

    # Lets normalize our data
    xdata = x_data/max(x_data)
    ydata = y_data/max(y_data)
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(sigmoid, xdata, ydata)
    #print the final parameters
    print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

    x = np.linspace(1960, 2015, 55)
    x = x/max(x)
    plt.figure(figsize=(8,5))
    y = sigmoid(x, *popt)
    plt.plot(xdata, ydata, 'ro', label='data')
    plt.plot(x, y, linewidth=3.0, label='fit')
    plt.legend(loc='best')
    plt.ylabel('GDP')
    plt.xlabel('Year')
    plt.show()



# createMask()
# LinearRegression()
# createMultiLine()
# predict_test()
# show()
# ploynomial()

# non_linear_regg()

non_linear_regg_inAct()