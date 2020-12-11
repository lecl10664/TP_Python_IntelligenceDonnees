#import

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# =============================================================================
# A.1
# =============================================================================

x = np.random.normal(loc = 0, scale = 1, size = 1000)
print(x)
s = pd.Series(x)
print(s)


# =============================================================================
# A.2
# =============================================================================
plt.title("Histogramme du bruit blanc gaussien contenant 1000 échantillons")
sns.distplot(x, hist=True)
plt.show()

plt.title("Bruit blanc gaussien contenant 1000 échantillons")
s.plot(figsize=(10,4))
plt.show()

# =============================================================================
# A.3
# =============================================================================
f = plt.figure(figsize=(12,8))
ax1 = f.add_subplot(211)
f = sm.graphics.tsa.plot_acf(s, lags=20, ax=ax1)
ax2 = f.add_subplot(212)
f = sm.graphics.tsa.plot_pacf(s, lags=20, ax=ax2)

# =============================================================================
# #A.4
# =============================================================================

#Test ADF
result = ts.adfuller(s, 1)
print(result)

def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = ts.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
adf_test(s)

#test KPSS

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = ts.kpss(timeseries, nlags='auto')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
    
kpss_test(x)


# =============================================================================
# B.1
# =============================================================================
sales = pd.read_csv("sales.csv")
food = sales['Food'][0:48]
fuel = sales['Fuel'][0:48]

print(food)

plt.title("Histogramme du chiffre d'affaire produits alimentaires ")
sns.distplot(food, hist=True)
plt.show()

plt.title("chiffre d'affaire produits alimentaires")
food.plot(figsize=(10,4))
plt.show()


fuel = sales['Fuel'][0:48]

plt.title("Histogramme du chiffre d'affaire de carburant ")
sns.distplot(fuel, hist=True)
plt.show()

plt.title("chiffre d'affaire d'affaire de carburant")
fuel.plot(figsize=(10,4))
plt.show()


# Test Shapiro-Wilk

shapiro_test_food = stats.shapiro(food)
print(shapiro_test_food)

shapiro_test_fuel = stats.shapiro(fuel)
print(shapiro_test_fuel)

# Test de Box-Pierce

res = sm.tsa.ARMA(food, (1,1)).fit(disp=-1)
sm.stats.acorr_ljungbox(res.resid, lags=10, boxpierce=True, return_df=True)
sm.stats.acorr_ljungbox(food, lags=10, boxpierce=True, return_df=True)


res = sm.tsa.ARMA(fuel, (1,1)).fit(disp=-1)
sm.stats.acorr_ljungbox(res.resid, return_df=True)
sm.stats.acorr_ljungbox(fuel,lags=10, boxpierce=True ,return_df=True)


f = plt.figure(figsize=(12,8))
ax1 = f.add_subplot(211)
f = sm.graphics.tsa.plot_acf(fuel, lags=20, ax=ax1)
ax2 = f.add_subplot(212)
f = sm.graphics.tsa.plot_pacf(fuel, lags=20, ax=ax2)


adf_test(fuel)
   
kpss_test(fuel)


# =============================================================================
# B.2
# =============================================================================

#1
fuel_d = fuel.diff()
fuel_d = fuel_d[1:]



#2
plt.title("diffrérentiation t - t+1 du chiffre d'affaire de carburant ")
s.plot(figsize=(10,4))
plt.show()



#3
f = plt.figure(figsize=(12,8))
ax1 = f.add_subplot(211)
f = sm.graphics.tsa.plot_acf(x, lags=20, ax=ax1)
ax2 = f.add_subplot(212)
f = sm.graphics.tsa.plot_pacf(x, lags=20, ax=ax2)

#4
adf_test(fuel_d)

kpss_test(fuel_d)


