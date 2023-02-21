import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
import pylab as pl
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

a = np.arange(20)
#print(a)

a = a.reshape(2,5,2)
#print(a)

b = np.random.randn(6,4)
#print(b)

dates = pd.date_range("20230101", periods=6)
df = pd.DataFrame(np.random.randn(6,4), dates)
# print (df)
# print(df[0])
# print(df.describe())
# print(df.head(2))

#d = np.arange(1,10, 0.2)
#plt.plot(d,d ** 2, 'r--')


data = {
    'a': np.arange(500),
    'b': np.random.randn(0,50,50),
    'c': np.random.randn(500)
}

#plt.scatter(data['a'], data['c'])
# plt.show()


df = pd.read_csv("FuelConsumption.csv")
# print(df.head())
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
# cdf.hist()
#plt.show()

def show():

    plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
    plt.xlabel('Eg size')
    plt.ylabel('Emssion')
    plt.show()

    sns.set_style(style="whitegrid")
    sns.pairplot(df)
    plt.show()

    plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
    plt.xlabel('Eg size')
    plt.ylabel('Emssion')
    plt.show()

train, test, regr = None
def createMask():
    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]
    # print(msk)
    # print([~msk])
    # print(train)
    # print(test)

def LinearRegression_Coefficients_Intercept():
    fig = plt.figure()
    pho1 = fig.add_subplot(111)
    pho1.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
    pho1.scatter(test.ENGINESIZE, test.CO2EMISSIONS, color='red')
    plt.xlabel('Eg size')
    plt.ylabel('Emssion')
    plt.show()

    regr = linear_model.LinearRegression()
    trainx = np.asanyarray(train[['ENGINESIZE']])
    trainy = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit(trainx, trainy)
    print('Coefficients:', regr.coef_)
    print('Intercept:', regr.intercept_)

    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
    plt.plot(trainx, regr.coef_[0][0] * trainx+regr.intercept_[0], '-r')
    #                          y = θ1 * X + θ0

    plt.xlabel('Eng size')
    plt.ylabel('Emission')
    plt.show()

def Evaloation():
    #----------------------------------------Evaloation
    testx = np.asanyarray(test[['ENGINESIZE']])
    testy = np.asanyarray(test[['CO2EMISSIONS']])
    test_y_ = regr.predict(testx)
    print('Main absolut error : %.2f' % np.mean(np.absolute(test_y_ - testy)))
    print('Residual sum of squares (MSE) : %.2f' % np.mean((test_y_ - testy) ** 2 ))
    print('r2_score : %.2f' % r2_score(testy, test_y_))

#---------------------------------------- multiple linear  --------------------------------------------

#MODELYEAR,MAKE,MODEL,VEHICLECLASS,ENGINESIZE,CYLINDERS,TRANSMISSION,FUELTYPE,FUELCONSUMPTION_CITY,FUELCONSUMPTION_HWY,FUELCONSUMPTION_COMB,FUELCONSUMPTION_COMB_MPG,CO2EMISSIONS
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',  'CO2EMISSIONS']]
def createMaskMultiLine():
    #print(cdf.head(9))
    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]
    regr = linear_model.LinearRegression()
    trainx = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
    trainy = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit(trainx, trainy)
    print('Coefficients:', regr.coef_)
    print('Intercept:', regr.intercept_)

    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
    plt.plot(trainx, regr.coef_[0][0]*trainx+regr.intercept_[0], '-r')
    #                          y = θ1 * X + θ0
    plt.xlabel('Eng size')
    plt.ylabel('Emission')
    plt.show()

#------------------------------predict test
def predict_test():
    y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
    x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
    y = np.asanyarray(test[['CO2EMISSIONS']])
    print("Residual sum of squares : %.2f" % np.mean((y_hat - y) ** 2))
    print('Variance score : %.2f' % regr.score(x, y))


#------------------------------polynomial
testx, testy, poly, clf = None
def polynomial():
    trainx = np.asanyarray(train[['ENGINESIZE']])
    trainy = np.asanyarray(train[['CO2EMISSIONS']])

    testx = np.asanyarray(test[['ENGINESIZE']])
    testy = np.asanyarray(test[['CO2EMISSIONS']])
    print(trainx[:3])

    poly = PolynomialFeatures(degree=2)
    trian_x_poly = poly.fit_transform(trainx)
    #print('train x poly:', trian_x_poly)

    clf = linear_model.LinearRegression()
    train_y_ = clf.fit(trian_x_poly, trainy)
    print('Coefficients:', clf.coef_)
    print('Intercept:', clf.intercept_)

    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
    xx = np.arange(0.0, 10.0, 0.1)
    yy = clf.intercept_[0] + clf.coef_[0][1] * xx + clf.coef_[0][2] * np.power(xx,2)
    plt.plot(xx, yy, '-r')
    #y = b + teta1 X + teta2 x**

    plt.xlabel('Eng size')
    plt.ylabel('Emission')
    plt.show()


    #----------------------------------------Evaloation
def Evaloation_polynomial():
    testx_poly = poly.fit_transform(testx)
    test_y_ = clf.predict(testx_poly)
    print('Main absolut error : %.2f' % np.mean(np.absolute(test_y_ - testy)))
    print('Residual sum of squares (MSE) : %.2f' % np.mean((test_y_ - testy) ** 2 ))
    print('R2 socre : %.2f' % r2_score(testy, test_y_))



#------------------------------Regg non linear reg
def non_linear_regg():
    x = np.arange(-5.0, 5.0, 0.1)
    y = 2 * (x) + 3
    y_noise = 2 * np.random.normal(size=x.size)
    ydata = y + y_noise

    plt.plot(x, ydata, 'bo')
    plt.plot(x,y, 'r')
    plt.xlabel('dep var')
    plt.ylabel('indep var')
    plt.show()

    #------------------------------Regg non linear reg example log(x)
    #x**3 + x**2 + x + 3
    x = np.arange(-5.0, 5.0, 0.1)
    y = 1 * (x**3) + 1 * (x**2) + 1 * x + 3
    y_noise = 20 * np.random.normal(size=x.size)
    ydata = y + y_noise

    plt.plot(x, ydata, 'bo')
    plt.plot(x,y, 'r')
    plt.xlabel('dep var')
    plt.ylabel('indep var')
    plt.show()


    #------------------------------Regg non linear reg example

    df = pd.read_csv("china_gdp.csv")
    plt.figure(figsize=(8,5))
    x_data, y_data = (df['Year'].values, df['Value'].values)
    plt.plot(x_data, y_data, 'ro')
    plt.xlabel('Year')
    plt.ylabel('GDP')
    plt.show()

