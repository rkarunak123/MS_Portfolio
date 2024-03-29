# -*- coding: utf-8 -*-
"""IST_718-Project_OLS.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_h45xdfwJ0Dy2LJ2q-Q20o9pbcQCeMLN

#Packages

list of packages that may have been used for presentation
"""

import pandas as pd
#from mlxtend.frequent_patterns import apriori
#from mlxtend.frequent_patterns import association_rules
#from mlxtend.preprocessing import TransactionEncoder

# important all packages that may work for this lab.

# import packages for analysis and modeling
import pandas as pd  # data frame operations
from pandas import Series
from pandas import DataFrame
#from pandas.tools.plotting import scatter_matrix  # scatter plot matrix

import numpy as np  # arrays and math functions
from scipy.stats import uniform  # for training-and-test split
import statsmodels.api as sm  # statistical models (including regression)
import statsmodels.formula.api as smf  # R-like model specification
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

from sklearn.tree import DecisionTreeRegressor  # machine learning tree
from sklearn.ensemble import RandomForestRegressor # ensemble method
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import boxcox
import matplotlib.pyplot as plt  # 2D plotting
import seaborn as sns  # PROVIDES TRELLIS AND SMALL MULTIPLE PLOTTING

"""#Data"""

#url = 'copied_raw_GH_link'
#df1 = pd.read_csv(url)

from google.colab import files
#uploaded = files.upload()
#CA_df=pd.read_csv('CAZipPopulations_healthcare_metro.csv')

url = 'https://raw.githubusercontent.com/mehrloo/IST718/master/Files%20for%20Submission/CAZipPopulations_healthcare_metro.csv?token=AMR6IBOYWZFGZFBLTUWVJPK5PUEOU'
CA_df = pd.read_csv(url)
df = pd.DataFrame(CA_df)


#checking data number of rows and columns

df.shape

df.head()

"""# Pre-Processing:

The first step in order to get the data in a format needed, some pre-processing was needed. Mainly, the month and year data in the csv file we split up horizotally, instead of vertically.

To fix the issue, I used the melt function.
"""

# A list of all the column names that will be used


col_names = ['Zip_Code',
'Metro',
'City',
'Population_2012',
'Home_Ownership_2012',
'Renter_2012',
'Poverty_Black_2012',
'Poverty_Hispanic_2012',
'Poverty_White_2012',
'Black_Insurance_Coverage_2012',
'Hispanic_Insurance_Coverage_2012',
'White_Insurance_Coverage_2012',
'Foreign_Born_2012',
'Total_Income_2012',
'Total_Income_Black_2012',
'Total_Income_Hispanic2012',
'Total_Income_White_2012',
'Population_White_2012',
'Population_Black_2012',
'Population_Asian_2012',
'Population_Other_2012',
'Population_2013',
'Home_Ownership_2013',
'Renter_2013',
'Poverty_Black_2013',
'Poverty_Hispanic_2013',
'Poverty_White_2013',
'Black_Insurance_Coverage_2013',
'Hispanic_Insurance_Coverage_2013',
'White_Insurance_Coverage_2013',
'Foreign_Born_2013',
'Total_Income_2013',
'Total_Income_Black_2013',
'Total_Income_Hispanic2013',
'Total_Income_White_2013',
'Population_White_2013',
'Population_Black_2013',
'Population_Asian_2013',
'Population_Other_2013',
'Population_2014',
'Home_Ownership_2014',
'Renter_2014',
'Poverty_Black_2014',
'Poverty_Hispanic_2014',
'Poverty_White_2014',
'Black_Insurance_Coverage_2014',
'Hispanic_Insurance_Coverage_2014',
'White_Insurance_Coverage_2014',
'Foreign_Born_2014',
'Total_Income_2014',
'Total_Income_Black_2014',
'Total_Income_Hispanic2014',
'Total_Income_White_2014',
'Population_White_2014',
'Population_Black_2014',
'Population_Asian_2014',
'Population_Other_2014',
'Population_2015',
'Home_Ownership_2015',
'Renter_2015',
'Poverty_Black_2015',
'Poverty_Hispanic_2015',
'Poverty_White_2015',
'Black_Insurance_Coverage_2015',
'Hispanic_Insurance_Coverage_2015',
'White_Insurance_Coverage_2015',
'Foreign_Born_2015',
'Total_Income_2015',
'Total_Income_Black_2015',
'Total_Income_Hispanic2015',
'Total_Income_White_2015',
'Population_White_2015',
'Population_Black_2015',
'Population_Asian_2015',
'Population_Other_2015',
'Population_2016',
'Home_Ownership_2016',
'Renter_2016',
'Poverty_Black_2016',
'Poverty_Hispanic_2016',
'Poverty_White_2016',
'Black_Insurance_Coverage_2016',
'Hispanic_Insurance_Coverage_2016',
'White_Insurance_Coverage_2016',
'Foreign_Born_2016',
'Total_Income_2016',
'Total_Income_Black_2016',
'Total_Income_Hispanic2016',
'Total_Income_White_2016',
'Population_White_2016',
'Population_Black_2016',
'Population_Asian_2016',
'Population_Other_2016',
'Population_2017',
'Home_Ownership_2017',
'Renter_2017',
'Poverty_Black_2017',
'Poverty_Hispanic_2017',
'Poverty_White_2017',
'Black_Insurance_Coverage_2017',
'Hispanic_Insurance_Coverage_2017',
'White_Insurance_Coverage_2017',
'Foreign_Born_2017',
'Total_Income_2017',
'Total_Income_Black_2017',
'Total_Income_Hispanic2017',
'Total_Income_White_2017',
'Population_White_2017',
'Population_Black_2017',
'Population_Asian_2017',
'Population_Other_2017'
]


# A list of all the column names that will be used


col_names = ['Zip_Code',
'Metro',
'City',
'Population_2012',
'Home_Ownership_2012',
'Renter_2012',
'Foreign_Born_2012',
'Poverty_Black_2012',
'Poverty_Hispanic_2012',
'Poverty_White_2012',
'Black_Insurance_Coverage_2012',
'Black_No_Insurance_Coverage_2012',            
'Hispanic_Insurance_Coverage_2012',
'Hispanic_No_Insurance_Coverage_2012',
'White_Insurance_Coverage_2012',
'White_No_Insurance_Coverage_2012',
'Total_Income_2012',
'Total_Income_Black_2012',
'Total_Income_Hispanic_2012',
'Total_Income_White_2012',
'Population_Black_2012',
'Population_White_2012',
'Population_Hispanic_2012',             
'Population_Other_2012',
'Population_2013',
'Home_Ownership_2013',
'Renter_2013',
'Foreign_Born_2013',
'Poverty_Black_2013',
'Poverty_Hispanic_2013',
'Poverty_White_2013',
'Black_Insurance_Coverage_2013',
'Black_No_Insurance_Coverage_2013',            
'Hispanic_Insurance_Coverage_2013',
'Hispanic_No_Insurance_Coverage_2013',
'White_Insurance_Coverage_2013',
'White_No_Insurance_Coverage_2013',
'Total_Income_2013',
'Total_Income_Black_2013',
'Total_Income_Hispanic_2013',
'Total_Income_White_2013',
'Population_Black_2013',
'Population_White_2013',
'Population_Hispanic_2013',   
'Population_Other_2013',
'Population_2014',
'Home_Ownership_2014',
'Renter_2014',
'Foreign_Born_2014',
'Poverty_Black_2014',
'Poverty_Hispanic_2014',
'Poverty_White_2014',
'Black_Insurance_Coverage_2014',
'Black_No_Insurance_Coverage_2014',            
'Hispanic_Insurance_Coverage_2014',
'Hispanic_No_Insurance_Coverage_2014',
'White_Insurance_Coverage_2014',
'White_No_Insurance_Coverage_2014',
'Total_Income_2014',
'Total_Income_Black_2014',
'Total_Income_Hispanic_2014',
'Total_Income_White_2014',
'Population_Black_2014',
'Population_White_2014',
'Population_Hispanic_2014',   
'Population_Other_2014',
'Population_2015',
'Home_Ownership_2015',
'Renter_2015',
'Foreign_Born_2015',
'Poverty_Black_2015',
'Poverty_Hispanic_2015',
'Poverty_White_2015',
'Black_Insurance_Coverage_2015',
'Black_No_Insurance_Coverage_2015',            
'Hispanic_Insurance_Coverage_2015',
'Hispanic_No_Insurance_Coverage_2015',
'White_Insurance_Coverage_2015',
'White_No_Insurance_Coverage_2015',
'Total_Income_2015',
'Total_Income_Black_2015',
'Total_Income_Hispanic_2015',
'Total_Income_White_2015',
'Population_Black_2015',
'Population_White_2015',
'Population_Hispanic_2015',   
'Population_Other_2015',
'Population_2016',
'Home_Ownership_2016',
'Renter_2016',
'Foreign_Born_2016',
'Poverty_Black_2016',
'Poverty_Hispanic_2016',
'Poverty_White_2016',
'Black_Insurance_Coverage_2016',
'Black_No_Insurance_Coverage_2016',            
'Hispanic_Insurance_Coverage_2016',
'Hispanic_No_Insurance_Coverage_2016',
'White_Insurance_Coverage_2016',
'White_No_Insurance_Coverage_2016',
'Total_Income_2016',
'Total_Income_Black_2016',
'Total_Income_Hispanic_2016',
'Total_Income_White_2016',
'Population_Black_2016',
'Population_White_2016',
'Population_Hispanic_2016',   
'Population_Other_2016',
'Population_2017',
'Home_Ownership_2017',
'Renter_2017',
'Foreign_Born_2017',
'Poverty_Black_2017',
'Poverty_Hispanic_2017',
'Poverty_White_2017',
'Black_Insurance_Coverage_2017',
'Black_No_Insurance_Coverage_2017',            
'Hispanic_Insurance_Coverage_2017',
'Hispanic_No_Insurance_Coverage_2017',
'White_Insurance_Coverage_2017',
'White_No_Insurance_Coverage_2017',
'Total_Income_2017',
'Total_Income_Black_2017',
'Total_Income_Hispanic_2017',
'Total_Income_White_2017',
'Population_Black_2017',
'Population_White_2017',
'Population_Hispanic_2017',   
'Population_Other_2017'
]

#using melt to transpose data
#using melt to transpose data
melted_df = pd.melt(df,id_vars = col_names,
                                   var_name = 'Date',
                                    value_name='Price')


#converting to Date format
melted_df['Date'] = pd.to_datetime(melted_df['Date'], format='%Y/%m/%d')

melted_df.shape

#Checking data
pd.DataFrame(melted_df.head())

"""#Visualization:
Note that this was NOT shown in presentation

##Boxlot
"""

#Graphic below shows salary histogram based on City for the last month available

melted_df_201712 = melted_df[(melted_df['Date'] > '2017-11-01')]

sns.boxplot(x="City", y="Price", data=melted_df_201712);
plt.ticklabel_format(style='plain', axis='y')

#Subsetting to some cities; and to 2012
df_CA_picked=melted_df[((melted_df['City']=='Los Angeles')|(melted_df['City']=='San Diego')|\
                       (melted_df['City']=='Irvine')|(melted_df['City']=='San Francisco')|(melted_df['City']=='Long Beach'))]
df_CA_picked = df_CA_picked[(df_CA_picked['Date'] > '2012-01-01') & (df_CA_picked['Date'] < '2013-01-01')]

df_CA_picked.shape # number of rows and columns


sns.boxplot(x="City", y="Price", data=df_CA_picked, width=.5);
plt.ticklabel_format(style='plain', axis='y')

#Subsetting to some cities; and to 201711
df_CA_picked=melted_df[((melted_df['City']=='Los Angeles')|(melted_df['City']=='San Diego')|\
                       (melted_df['City']=='Irvine')|(melted_df['City']=='San Francisco')|(melted_df['City']=='Long Beach'))]
df_CA_picked = df_CA_picked[(df_CA_picked['Date'] > '2017-01-01')]

df_CA_picked.shape # number of rows and columns


sns.boxplot(x="City", y="Price", data=df_CA_picked, width=.5);
plt.ticklabel_format(style='plain', axis='y')

#Subsetting to some cities and just 2017
df_CA_picked2=melted_df[((melted_df['City']=='Modesto')|(melted_df['City']=='Pasadena')|(melted_df['City']=='Glendale')|(melted_df['City']=='Anaheim')|(melted_df['City']=='Berkeley'))]
df_CA_picked2 = df_CA_picked2[(df_CA_picked2['Date'] > '2017-01-01')]

df_CA_picked2.shape # number of rows and columns


sns.boxplot(x="City", y="Price", data=df_CA_picked2, width=.5);
plt.ticklabel_format(style='plain', axis='y')

df_CA_picked_viz=df_CA_picked[((df_CA_picked['City']=='Los Angeles'))]

sns.boxplot(x="City", y="Price", data=df_CA_picked_viz);
plt.ticklabel_format(style='plain', axis='y')

df_CA_picked_viz=df_CA_picked[((df_CA_picked['City']=='San Diego'))]

sns.boxplot(x="City", y="Price", data=df_CA_picked_viz);
plt.ticklabel_format(style='plain', axis='y')

df_CA_picked_viz=df_CA_picked[((df_CA_picked['City']=='Long Beach'))]

sns.boxplot(x="City", y="Price", data=df_CA_picked_viz);
plt.ticklabel_format(style='plain', axis='y')

"""#Model; All Variables
Note that this was NOT shown in presentation
"""

#recall Data
model_df = melted_df.drop(columns = ['Zip_Code','Metro','Date'])

#model_df.shape
model_df.head()

est = smf.ols(formula='Price ~ City+Population_2012+Home_Ownership_2012+Renter_2012+Poverty_Black_2012+Poverty_Hispanic_2012+Poverty_White_2012+Black_Insurance_Coverage_2012+Hispanic_Insurance_Coverage_2012+White_Insurance_Coverage_2012+Foreign_Born_2012+Total_Income_2012+Total_Income_Black_2012+Total_Income_Hispanic_2012+Total_Income_White_2012+Population_White_2012+Population_Black_2012+Population_Hispanic_2012+Population_Other_2012+Population_2013+Home_Ownership_2013+Renter_2013+Poverty_Black_2013+Poverty_Hispanic_2013+Poverty_White_2013+Black_Insurance_Coverage_2013+Hispanic_Insurance_Coverage_2013+White_Insurance_Coverage_2013+Foreign_Born_2013+Total_Income_2013+Total_Income_Black_2013+Total_Income_Hispanic_2013+Total_Income_White_2013+Population_White_2013+Population_Black_2013+Population_Hispanic_2013+Population_Other_2013+Population_2014+Home_Ownership_2014+Renter_2014+Poverty_Black_2014+Poverty_Hispanic_2014+Poverty_White_2014+Black_Insurance_Coverage_2014+Hispanic_Insurance_Coverage_2014+White_Insurance_Coverage_2014+Foreign_Born_2014+Total_Income_2014+Total_Income_Black_2014+Total_Income_Hispanic_2014+Total_Income_White_2014+Population_White_2014+Population_Black_2014+Population_Hispanic_2014+Population_Other_2014+Population_2015+Home_Ownership_2015+Renter_2015+Poverty_Black_2015+Poverty_Hispanic_2015+Poverty_White_2015+Black_Insurance_Coverage_2015+Hispanic_Insurance_Coverage_2015+White_Insurance_Coverage_2015+Foreign_Born_2015+Total_Income_2015+Total_Income_Black_2015+Total_Income_Hispanic_2015+Total_Income_White_2015+Population_White_2015+Population_Black_2015+Population_Hispanic_2015+Population_Other_2015+Population_2016+Home_Ownership_2016+Renter_2016+Poverty_Black_2016+Poverty_Hispanic_2016+Poverty_White_2016+Black_Insurance_Coverage_2016+Hispanic_Insurance_Coverage_2016+White_Insurance_Coverage_2016+Foreign_Born_2016+Total_Income_2016+Total_Income_Black_2016+Total_Income_Hispanic_2016+Total_Income_White_2016+Population_White_2016+Population_Black_2016+Population_Hispanic_2016+Population_Other_2016+Population_2017+Home_Ownership_2017+Renter_2017+Poverty_Black_2017+Poverty_Hispanic_2017+Poverty_White_2017+Black_Insurance_Coverage_2017+Hispanic_Insurance_Coverage_2017+White_Insurance_Coverage_2017+Foreign_Born_2017+Total_Income_2017+Total_Income_Black_2017+Total_Income_Hispanic_2017+Total_Income_White_2017+Population_White_2017+Population_Black_2017+Population_Hispanic_2017+Population_Other_2017'
              , data=model_df).fit()
est.summary()

"""# Model; City/Year Specific;
Models below were shown in presentation

###City Specific; Los Angeles
"""

# dropping columns that are not needed
model_df = melted_df.drop(columns = ['Zip_Code','Metro','Date'])

model_df.shape

df_CA_picked_LA = model_df[((model_df['City']=='Los Angeles'))]
df_CA_picked_LA = df_CA_picked_LA.drop(columns = [])
model_df=df_CA_picked_LA
#est = smf.ols(formula='Price ~ City+Population_2012+Home_Ownership_2012+Renter_2012+Poverty_Black_2012+Poverty_Hispanic_2012+Poverty_White_2012+Black_Insurance_Coverage_2012+Hispanic_Insurance_Coverage_2012+White_Insurance_Coverage_2012+Foreign_Born_2012+Total_Income_2012+Total_Income_Black_2012+Total_Income_Hispanic_2012+Total_Income_White_2012+Population_White_2012+Population_Black_2012+Population_Hispanic_2012+Population_Other_2012+Population_2013+Home_Ownership_2013+Renter_2013+Poverty_Black_2013+Poverty_Hispanic_2013+Poverty_White_2013+Black_Insurance_Coverage_2013+Hispanic_Insurance_Coverage_2013+White_Insurance_Coverage_2013+Foreign_Born_2013+Total_Income_2013+Total_Income_Black_2013+Total_Income_Hispanic_2013+Total_Income_White_2013+Population_White_2013+Population_Black_2013+Population_Hispanic_2013+Population_Other_2013+Population_2014+Home_Ownership_2014+Renter_2014+Poverty_Black_2014+Poverty_Hispanic_2014+Poverty_White_2014+Black_Insurance_Coverage_2014+Hispanic_Insurance_Coverage_2014+White_Insurance_Coverage_2014+Foreign_Born_2014+Total_Income_2014+Total_Income_Black_2014+Total_Income_Hispanic_2014+Total_Income_White_2014+Population_White_2014+Population_Black_2014+Population_Hispanic_2014+Population_Other_2014+Population_2015+Home_Ownership_2015+Renter_2015+Poverty_Black_2015+Poverty_Hispanic_2015+Poverty_White_2015+Black_Insurance_Coverage_2015+Hispanic_Insurance_Coverage_2015+White_Insurance_Coverage_2015+Foreign_Born_2015+Total_Income_2015+Total_Income_Black_2015+Total_Income_Hispanic_2015+Total_Income_White_2015+Population_White_2015+Population_Black_2015+Population_Hispanic_2015+Population_Other_2015+Population_2016+Home_Ownership_2016+Renter_2016+Poverty_Black_2016+Poverty_Hispanic_2016+Poverty_White_2016+Black_Insurance_Coverage_2016+Hispanic_Insurance_Coverage_2016+White_Insurance_Coverage_2016+Foreign_Born_2016+Total_Income_2016+Total_Income_Black_2016+Total_Income_Hispanic_2016+Total_Income_White_2016+Population_White_2016+Population_Black_2016+Population_Hispanic_2016+Population_Other_2016+Population_2017+Home_Ownership_2017+Renter_2017+Poverty_Black_2017+Poverty_Hispanic_2017+Poverty_White_2017+Black_Insurance_Coverage_2017+Hispanic_Insurance_Coverage_2017+White_Insurance_Coverage_2017+Foreign_Born_2017+Total_Income_2017+Total_Income_Black_2017+Total_Income_Hispanic_2017+Total_Income_White_2017+Population_White_2017+Population_Black_2017+Population_Hispanic_2017+Population_Other_2017'
est = smf.ols(formula='Price ~ Population_2012+Home_Ownership_2012+Renter_2012+Poverty_Black_2012+Poverty_Hispanic_2012+Poverty_White_2012+Black_Insurance_Coverage_2012+Hispanic_Insurance_Coverage_2012+White_Insurance_Coverage_2012+Foreign_Born_2012+Total_Income_2012+Total_Income_Black_2012+Total_Income_Hispanic_2012+Total_Income_White_2012+Population_White_2012+Population_Black_2012+Population_Hispanic_2012+Population_Other_2012'
              , data=model_df).fit()
est.summary()

"""###City Specific; San Francisco"""

# dropping coulumns that are not needed
model_df = melted_df.drop(columns = ['Zip_Code','Metro','Date'])

model_df.shape

df_CA_picked_LA = model_df[((model_df['City']=='San Francisco'))]
model_df=df_CA_picked_LA

#est = smf.ols(formula='Price ~ City+Population_2012+Home_Ownership_2012+Renter_2012+Poverty_Black_2012+Poverty_Hispanic_2012+Poverty_White_2012+Black_Insurance_Coverage_2012+Hispanic_Insurance_Coverage_2012+White_Insurance_Coverage_2012+Foreign_Born_2012+Total_Income_2012+Total_Income_Black_2012+Total_Income_Hispanic_2012+Total_Income_White_2012+Population_White_2012+Population_Black_2012+Population_Hispanic_2012+Population_Other_2012+Population_2013+Home_Ownership_2013+Renter_2013+Poverty_Black_2013+Poverty_Hispanic_2013+Poverty_White_2013+Black_Insurance_Coverage_2013+Hispanic_Insurance_Coverage_2013+White_Insurance_Coverage_2013+Foreign_Born_2013+Total_Income_2013+Total_Income_Black_2013+Total_Income_Hispanic_2013+Total_Income_White_2013+Population_White_2013+Population_Black_2013+Population_Hispanic_2013+Population_Other_2013+Population_2014+Home_Ownership_2014+Renter_2014+Poverty_Black_2014+Poverty_Hispanic_2014+Poverty_White_2014+Black_Insurance_Coverage_2014+Hispanic_Insurance_Coverage_2014+White_Insurance_Coverage_2014+Foreign_Born_2014+Total_Income_2014+Total_Income_Black_2014+Total_Income_Hispanic_2014+Total_Income_White_2014+Population_White_2014+Population_Black_2014+Population_Hispanic_2014+Population_Other_2014+Population_2015+Home_Ownership_2015+Renter_2015+Poverty_Black_2015+Poverty_Hispanic_2015+Poverty_White_2015+Black_Insurance_Coverage_2015+Hispanic_Insurance_Coverage_2015+White_Insurance_Coverage_2015+Foreign_Born_2015+Total_Income_2015+Total_Income_Black_2015+Total_Income_Hispanic_2015+Total_Income_White_2015+Population_White_2015+Population_Black_2015+Population_Hispanic_2015+Population_Other_2015+Population_2016+Home_Ownership_2016+Renter_2016+Poverty_Black_2016+Poverty_Hispanic_2016+Poverty_White_2016+Black_Insurance_Coverage_2016+Hispanic_Insurance_Coverage_2016+White_Insurance_Coverage_2016+Foreign_Born_2016+Total_Income_2016+Total_Income_Black_2016+Total_Income_Hispanic_2016+Total_Income_White_2016+Population_White_2016+Population_Black_2016+Population_Hispanic_2016+Population_Other_2016+Population_2017+Home_Ownership_2017+Renter_2017+Poverty_Black_2017+Poverty_Hispanic_2017+Poverty_White_2017+Black_Insurance_Coverage_2017+Hispanic_Insurance_Coverage_2017+White_Insurance_Coverage_2017+Foreign_Born_2017+Total_Income_2017+Total_Income_Black_2017+Total_Income_Hispanic_2017+Total_Income_White_2017+Population_White_2017+Population_Black_2017+Population_Hispanic_2017+Population_Other_2017'
est = smf.ols(formula='Price ~ City+Population_2012+Home_Ownership_2012+Renter_2012+Poverty_Black_2012+Poverty_Hispanic_2012+Poverty_White_2012+Black_Insurance_Coverage_2012+Hispanic_Insurance_Coverage_2012+White_Insurance_Coverage_2012+Foreign_Born_2012+Total_Income_2012+Total_Income_Black_2012+Total_Income_Hispanic_2012+Total_Income_White_2012+Population_White_2012+Population_Black_2012+Population_Hispanic_2012+Population_Other_2012'
              , data=model_df).fit()
est.summary()

"""#Model; with Dummy Variables
Note that this was NOT shown in presentation

##Data
"""

#recall Data
model_df = melted_df.drop(columns = ['Zip_Code','Metro','Date'])

#model_df.shape
model_df.head()

"""##Variables Manipulation"""

#Recall Data

model_df2 = melted_df

"""####Native or Foreign Born"""

def f(row):
    if row['Population_2014']/2 < row['Foreign_Born_2014']:
        val = 'ForeignBorn'
    else:
        val = 'NativeBorn'
    return val

model_df2['PopType'] = model_df2.apply(f, axis=1)

"""####IncomeBraket"""

def f(row):
    if row['Total_Income_2014'] < 30000:
        val = 'Not_Upper_Income'
    else:
        val = 'Upper_Income'
    return val

model_df2['Incomebracket'] = model_df2.apply(f, axis=1)
model_df2=model_df2[model_df2['Renter_2017']>0]
model_df2=model_df2[model_df2['Renter_2012']>0]
#model_df2=model_df2[['Zip_Code','Total_Income_2017']]

"""####Ownership"""

def f(row):
  #if(row['Renter_2017']>0)&row['Renter_2012'])
    if ((row['Home_Ownership_2017']/row['Renter_2017'])<(row['Home_Ownership_2012']/row['Renter_2012'])):
        val = 'HomeIncrease'
    else:
        val = 'RentIncrease'
    return val

model_df2['Ownership'] = model_df2.apply(f, axis=1)

"""##Modeling, Second"""

model_df2.shape

model_df2.head(1)

#Creating Dummy Variables

df_Ownership = pd.get_dummies(model_df2['Ownership'])
df_Incomebracket = pd.get_dummies(model_df2['Incomebracket'])
df_PopType = pd.get_dummies(model_df2['PopType'])

model_df2 = pd.concat([model_df2, df_Ownership], axis=1)
model_df2 = pd.concat([model_df2, df_Incomebracket], axis=1)
model_df2 = pd.concat([model_df2, df_PopType], axis=1)




model_df2.head(1)

# We are going to drop all the categorical Columns

#model_df2 = melted_df.drop(columns = ['Zip_Code','Metro','Population_2012','Home_Ownership_2012','Renter_2012','Poverty_Black_2012','Poverty_Hispanic_2012','Poverty_White_2012','Black_Insurance_Coverage_2012','Hispanic_Insurance_Coverage_2012','White_Insurance_Coverage_2012','Foreign_Born_2012','Total_Income_2012','Total_Income_Black_2012','Total_Income_Hispanic2012','Total_Income_White_2012','Population_White_2012','Population_Black_2012','Population_Asian_2012','Population_Other_2012','Population_2013','Home_Ownership_2013','Renter_2013','Poverty_Black_2013','Poverty_Hispanic_2013','Poverty_White_2013','Black_Insurance_Coverage_2013','Hispanic_Insurance_Coverage_2013','White_Insurance_Coverage_2013','Foreign_Born_2013','Total_Income_2013','Total_Income_Black_2013','Total_Income_Hispanic2013','Total_Income_White_2013','Population_White_2013','Population_Black_2013','Population_Asian_2013','Population_Other_2013','Population_2014','Home_Ownership_2014','Renter_2014','Poverty_Black_2014','Poverty_Hispanic_2014','Poverty_White_2014','Black_Insurance_Coverage_2014','Hispanic_Insurance_Coverage_2014','White_Insurance_Coverage_2014','Foreign_Born_2014','Total_Income_2014','Total_Income_Black_2014','Total_Income_Hispanic2014','Total_Income_White_2014','Population_White_2014','Population_Black_2014','Population_Asian_2014','Population_Other_2014','Population_2015','Home_Ownership_2015','Renter_2015','Poverty_Black_2015','Poverty_Hispanic_2015','Poverty_White_2015','Black_Insurance_Coverage_2015','Hispanic_Insurance_Coverage_2015','White_Insurance_Coverage_2015','Foreign_Born_2015','Total_Income_2015','Total_Income_Black_2015','Total_Income_Hispanic2015','Total_Income_White_2015','Population_White_2015','Population_Black_2015','Population_Asian_2015','Population_Other_2015','Population_2016','Home_Ownership_2016','Renter_2016','Poverty_Black_2016','Poverty_Hispanic_2016','Poverty_White_2016','Black_Insurance_Coverage_2016','Hispanic_Insurance_Coverage_2016','White_Insurance_Coverage_2016','Foreign_Born_2016','Total_Income_2016','Total_Income_Black_2016','Total_Income_Hispanic2016','Total_Income_White_2016','Population_White_2016','Population_Black_2016','Population_Asian_2016','Population_Other_2016','Population_2017','Home_Ownership_2017','Renter_2017','Poverty_Black_2017','Poverty_Hispanic_2017','Poverty_White_2017','Black_Insurance_Coverage_2017','Hispanic_Insurance_Coverage_2017','White_Insurance_Coverage_2017','Foreign_Born_2017','Total_Income_2017','Total_Income_Black_2017','Total_Income_Hispanic2017','Total_Income_White_2017','Population_White_2017','Population_Black_2017','Population_Asian_2017','Population_Other_2017','Ownership','Incomebracket','PopType'])
#model_df2 = model_df2.drop(columns = ['Zip_Code','Metro','Population_2012','Home_Ownership_2012','Renter_2012','Poverty_Black_2012','Poverty_Hispanic_2012','Poverty_White_2012','Black_Insurance_Coverage_2012','Hispanic_Insurance_Coverage_2012','White_Insurance_Coverage_2012','Foreign_Born_2012','Total_Income_2012','Total_Income_Black_2012','Total_Income_Hispanic_2012','Total_Income_White_2012','Population_White_2012','Population_Black_2012','Population_Hispanic_2012','Population_Other_2012','Population_2013','Home_Ownership_2013','Renter_2013','Poverty_Black_2013','Poverty_Hispanic_2013','Poverty_White_2013','Black_Insurance_Coverage_2013','Hispanic_Insurance_Coverage_2013','White_Insurance_Coverage_2013','Foreign_Born_2013','Total_Income_2013','Total_Income_Black_2013','Total_Income_Hispanic_2013','Total_Income_White_2013','Population_White_2013','Population_Black_2013','Population_Hispanic_2013','Population_Other_2013','Population_2014','Home_Ownership_2014','Renter_2014','Poverty_Black_2014','Poverty_Hispanic_2014','Poverty_White_2014','Black_Insurance_Coverage_2014','Hispanic_Insurance_Coverage_2014','White_Insurance_Coverage_2014','Foreign_Born_2014','Total_Income_2014','Total_Income_Black_2014','Total_Income_Hispanic_2014','Total_Income_White_2014','Population_White_2014','Population_Black_2014','Population_Hispanic_2014','Population_Other_2014','Population_2015','Home_Ownership_2015','Renter_2015','Poverty_Black_2015','Poverty_Hispanic_2015','Poverty_White_2015','Black_Insurance_Coverage_2015','Hispanic_Insurance_Coverage_2015','White_Insurance_Coverage_2015','Foreign_Born_2015','Total_Income_2015','Total_Income_Black_2015','Total_Income_Hispanic_2015','Total_Income_White_2015','Population_White_2015','Population_Black_2015','Population_Hispanic_2015','Population_Other_2015','Population_2016','Home_Ownership_2016','Renter_2016','Poverty_Black_2016','Poverty_Hispanic_2016','Poverty_White_2016','Black_Insurance_Coverage_2016','Hispanic_Insurance_Coverage_2016','White_Insurance_Coverage_2016','Foreign_Born_2016','Total_Income_2016','Total_Income_Black_2016','Total_Income_Hispanic_2016','Total_Income_White_2016','Population_White_2016','Population_Black_2016','Population_Hispanic_2016','Population_Other_2016','Population_2017','Home_Ownership_2017','Renter_2017','Poverty_Black_2017','Poverty_Hispanic_2017','Poverty_White_2017','Black_Insurance_Coverage_2017','Hispanic_Insurance_Coverage_2017','White_Insurance_Coverage_2017','Foreign_Born_2017','Total_Income_2017','Total_Income_Black_2017','Total_Income_Hispanic_2017','Total_Income_White_2017','Population_White_2017','Population_Black_2017','Population_Hispanic_2017','Population_Other_2017'])
model_df2 = model_df2.drop(columns = ['Zip_Code','Metro','Population_2012','Home_Ownership_2012','Renter_2012','Poverty_Black_2012','Poverty_Hispanic_2012','Poverty_White_2012','Black_Insurance_Coverage_2012','Hispanic_Insurance_Coverage_2012','White_Insurance_Coverage_2012','Foreign_Born_2012','Total_Income_2012','Total_Income_Black_2012','Total_Income_Hispanic_2012','Total_Income_White_2012','Population_White_2012','Population_Black_2012','Population_Hispanic_2012','Population_Other_2012','Population_2013','Home_Ownership_2013','Renter_2013','Poverty_Black_2013','Poverty_Hispanic_2013','Poverty_White_2013','Black_Insurance_Coverage_2013','Hispanic_Insurance_Coverage_2013','White_Insurance_Coverage_2013','Foreign_Born_2013','Total_Income_2013','Total_Income_Black_2013','Total_Income_Hispanic_2013','Total_Income_White_2013','Population_White_2013','Population_Black_2013','Population_Hispanic_2013','Population_Other_2013','Population_2014','Home_Ownership_2014','Renter_2014','Poverty_Black_2014','Poverty_Hispanic_2014','Poverty_White_2014','Black_Insurance_Coverage_2014','Hispanic_Insurance_Coverage_2014','White_Insurance_Coverage_2014','Foreign_Born_2014','Total_Income_2014','Total_Income_Black_2014','Total_Income_Hispanic_2014','Total_Income_White_2014','Population_White_2014','Population_Black_2014','Population_Hispanic_2014','Population_Other_2014','Population_2015','Home_Ownership_2015','Renter_2015','Poverty_Black_2015','Poverty_Hispanic_2015','Poverty_White_2015','Black_Insurance_Coverage_2015','Hispanic_Insurance_Coverage_2015','White_Insurance_Coverage_2015','Foreign_Born_2015','Total_Income_2015','Total_Income_Black_2015','Total_Income_Hispanic_2015','Total_Income_White_2015','Population_White_2015','Population_Black_2015','Population_Hispanic_2015','Population_Other_2015','Population_2016','Home_Ownership_2016','Renter_2016','Poverty_Black_2016','Poverty_Hispanic_2016','Poverty_White_2016','Black_Insurance_Coverage_2016','Hispanic_Insurance_Coverage_2016','White_Insurance_Coverage_2016','Foreign_Born_2016','Total_Income_2016','Total_Income_Black_2016','Total_Income_Hispanic_2016','Total_Income_White_2016','Population_White_2016','Population_Black_2016','Population_Hispanic_2016','Population_Other_2016','Population_2017','Home_Ownership_2017','Renter_2017','Poverty_Black_2017','Poverty_Hispanic_2017','Poverty_White_2017','Black_Insurance_Coverage_2017','Hispanic_Insurance_Coverage_2017','White_Insurance_Coverage_2017','Foreign_Born_2017','Total_Income_2017','Total_Income_Black_2017','Total_Income_Hispanic_2017','Total_Income_White_2017','Population_White_2017','Population_Black_2017','Population_Hispanic_2017','Population_Other_2017','Black_No_Insurance_Coverage_2012', 'Hispanic_No_Insurance_Coverage_2012','White_No_Insurance_Coverage_2012','Black_No_Insurance_Coverage_2013','Hispanic_No_Insurance_Coverage_2013','White_No_Insurance_Coverage_2013','Black_No_Insurance_Coverage_2014','Hispanic_No_Insurance_Coverage_2014','White_No_Insurance_Coverage_2014','Black_No_Insurance_Coverage_2015','Hispanic_No_Insurance_Coverage_2015','White_No_Insurance_Coverage_2015','Black_No_Insurance_Coverage_2016','Hispanic_No_Insurance_Coverage_2016','White_No_Insurance_Coverage_2016','Black_No_Insurance_Coverage_2017','Hispanic_No_Insurance_Coverage_2017','White_No_Insurance_Coverage_2017','PopType','Incomebracket','Ownership'])

model_df2.shape

model_df2.head()

model_df3 = model_df2

#colss = [21,22,23]
#model_df3.drop(model_df3.columns[colss],axis=1,inplace=True)

#model_df3.shape

model_df3.head(1)

model_df3 = model_df2
#model_df3 = model_df3[((model_df3['City']=='San Francisco'))]


est = smf.ols(formula='Price ~ City + HomeIncrease + RentIncrease + Not_Upper_Income + Upper_Income + ForeignBorn + NativeBorn'
              , data=model_df3).fit()
est.summary()