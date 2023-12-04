# BigMacIndex

# Overview
This repository contains Python code for analyzing the relationship between Big Mac prices and hourly wages using linear regression. The code is implemented in Google Colab and utilizes various libraries such as pandas, matplotlib, seaborn, and statsmodels.

# Instructions
1. Mount Google Drive:
* Ensure you have your data stored in Google Drive.
  
  from google.colab import drive
  drive.mount('/content/drive/')
  import os
  os.chdir('/content/drive/MyDrive/Data Science ')

2. Load Data:
Load the Big Mac Index data from an Excel file into a Pandas DataFrame.
python
import pandas as pd
df = pd.read_excel("/content/drive/MyDrive/Data Science /Big Mac Index

3. Data Visualization:
Visualize the relationship between 'Big_Mac_Price' and 'Hourly_Wages' using a scatter plot.
python
import matplotlib.pyplot as plt
import seaborn as sns
sns.relplot(x='Big_Mac_Price', y='Hourly_Wages', data=df)
plt.show()

4. Linear Regression:
Fit a linear regression model using the Ordinary Least Squares (OLS) method from the statsmodels library.
python
import statsmodels.api as sm

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
ssr = np.sum(np.square(net_hourly_wages_pred - net_hourly_wages_pred.mean()))
sse = np.sum(np.square(net_hourly_wages_pred - df['Hourly_Wages'].values))

df_ssr = 1
df_sse = 27 - 1 - df_ssr

F_stats = (ssr/df_ssr)/(sse/df_sse)
r_square = 1 - (sse / (sse + ssr))

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

