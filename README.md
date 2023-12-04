# BigMacIndex

# Overview
This repository contains Python code for analyzing the relationship between Big Mac prices(Independent Variable) and hourly wages(Dependent Variable) using linear regression. The code is implemented in Google Colab and utilizes various libraries such as pandas, matplotlib, seaborn, and statsmodels.
Where I have answer to a following question

1. Is there a relationship between the price od a Big Mac and the net hourly wages of workers around the world? Of so , How strong is the Relationship ?

2. Is it possible to develop a model to predict or determine the net hourly wage of a worker around the world by the price of a Big Mac hamburger in that country? If so , how good is the model?

3. if a model can be constructed to determine the net hourly wage of a worker around the world by the price of a Big Mac hamburger, what would be the predicted net hourly wage of a worker in a country if the price of a Big Mac Hamburger was $3.00?

# Instructions
1. Mount Google Drive:
* Ensure you have your data stored in Google Drive.
  
  from google.colab import drive
  drive.mount('/content/drive/')
  import os
  os.chdir('/content/drive/MyDrive/Data Science ')

2. Load Data:
Big Mac Index data has Big Mac prices and Net hourly wages of 27 countries
Note : net hourly wages are based on a weighted average of 12 professional 
Load the Big Mac Index data from an Excel file into a Pandas DataFrame.
python
import pandas as pd
df = pd.read_excel("/content/drive/MyDrive/Data Science /Big Mac Index

4. Data Visualization:
Visualize the relationship between 'Big_Mac_Price' and 'Hourly_Wages' using a scatter plot.
python
import matplotlib.pyplot as plt
import seaborn as sns
sns.relplot(x='Big_Mac_Price', y='Hourly_Wages', data=df)
plt.show()

5. Linear Regression:
Fit a linear regression model using the Ordinary Least Squares (OLS) method from the statsmodels library.
python
note: OLS is a method used to estimate the parameters in a linear regression model. It minimizes the sum of squared differences between the observed and predicted values, providing the "best-fitting" line.
import statsmodels.api as sm
Note : statsmodels.api is used to perform linear regression analysis with the Ordinary Least Squares (OLS) method. The statsmodels library is a Python package that provides classes and functions for estimating and testing statistical models.
model = sm.OLS.from_formula('Hourly_Wages ~ Big_Mac_Price', data=df)
result = model.fit()
print(result.summary())

5. Plot Regression Line:
Plot the regression line on the scatter plot.
net_hourly_wages_pred = -4.5397 +  4.7435 * df['Big_Mac_Price']

sns.relplot(x='Big_Mac_Price', y='Hourly_Wages', data=df)
plt.plot(df['Big_Mac_Price'], net_hourly_wages_pred, 'r-')
plt.show()

6. ANOVA and R-squared Calculation:
Calculate ANOVA F-statistics and R-squared.
Note: ANOVA (Analysis of Variance) F-statistics and R-squared are calculated in the context of linear regression to assess the overall significance of the regression model and to quantify the proportion of variance in the dependent variable explained by the independent variables.
sum of squares due to regression (SSR) by the degrees of freedom associated with regression (df_sse).

ssr = np.sum(np.square(net_hourly_wages_pred - net_hourly_wages_pred.mean()))
sse = np.sum(np.square(net_hourly_wages_pred - df['Hourly_Wages'].values))

df_ssr = 1
df_sse = 27 - 1 - df_ssr

F_stats = (ssr/df_ssr)/(sse/df_sse)
r_square = 1 - (sse / (sse + ssr))
Note : F_stats 

print('F Statistics =', F_stats)
print('R Square Value =', r_square)

7. Correlation Analysis:
Calculate the correlation matrix for the dataset.
df.corr()
Square the correlation coefficient for interpretation.
np.square(0.813396)

8. Make Predictions:
Make predictions using the fitted regression model.
net_hourly_wages_pred = -4.5397 +  4.7435 * 3

