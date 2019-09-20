# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:48:21 2019

@author: Rebecca Karunakaran
"""

from pandas import Series
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from math import sqrt
from pandas import DataFrame
from scipy.stats import boxcox
from pandas import concat
from numpy import mean
from fbprophet import Prophet

#-----------------------------------------------------------------------------------
#Load the Data
#------------------------------------------------------------------------------------ 

Zillow_df = pd.read_csv('C:\Syracuse\Zip_Zhvi_SingleFamilyResidence.csv', header=0)


#-----------------------------------------------------------------------------------
#Review the Data and Clean as needed
#------------------------------------------------------------------------------------
# print the first five rows of the data frame
print(pd.DataFrame.head(Zillow_df))

#Check for null of missing values and replace with 0s
Zillow_df = Zillow_df.fillna(0)

#Check for duplicate zipcodes
print("Checking for duplicate data")
print(any(Zillow_df['RegionName'].duplicated()))
print(any(Zillow_df['RegionID'].duplicated()))


#---------------------------------------------------------------------------------
#Explore the data
#--------------------------------------------------------------------------------

#Time Series Plot for Hot Springs, Arkansas metro
HotSprings_df =  Zillow_df[Zillow_df.Metro == 'Hot Springs']

#Calc Average
HotSprings_df.loc['Average'] =  HotSprings_df.mean()

#flip the data
HotSprings_df_t = HotSprings_df.transpose()
HotSprings_df_transposed= HotSprings_df_t.iloc[7:]
HotSprings_df_transposed= HotSprings_df_transposed.drop(HotSprings_df_transposed.columns[:4], axis=1)

#Create the series
HotSprings_series = Series(HotSprings_df_transposed['Average'], dtype='int32')

print('Hot Springs')
print(HotSprings_series.describe())

#Plot the series
from matplotlib import pyplot
HotSprings_series.plot()
pyplot.title('Time Series Plot for House Prices in Hot Springs Arkansas') 
pyplot.show()

#--------------------------------------------------------------
#Time Series Plot for Little Rock, Arkansas metro
LittleRock_df =  Zillow_df[Zillow_df.Metro == 'Little Rock-North Little Rock-Conway']

#Calc Average
LittleRock_df.loc['Average'] =  LittleRock_df.mean()

#flip the data
LittleRock_df_t = LittleRock_df.transpose()
LittleRock_df_transposed= LittleRock_df_t.iloc[7:]
LittleRock_df_transposed= LittleRock_df_transposed.drop(LittleRock_df_transposed.columns[:4], axis=1)

#Create the series
HotSprings_series = Series(LittleRock_df_transposed['Average'], dtype='int32')

print('Hot Springs')
print(HotSprings_series.describe())

#Plot the series
from matplotlib import pyplot
HotSprings_series.plot()
pyplot.title('Time Series Plot for House Prices in Little Rock, Arkansas') 
pyplot.show()

#--------------------------------------------------------------
#Time Series Plot for Fayetteville, Arkansas metro
Fayetteville_df =  Zillow_df[Zillow_df['Metro'].isin(['Fayetteville', 'Fayetteville-Springdale-Rogers'])]

#Calc Average
Fayetteville_df.loc['Average'] =  Fayetteville_df.mean()

#flip the data
Fayetteville_df_t = Fayetteville_df.transpose()
Fayetteville_df_transposed= Fayetteville_df_t.iloc[7:]
Fayetteville_df_transposed= Fayetteville_df_transposed.drop(Fayetteville_df_transposed.columns[:4], axis=1)

#Create the series
HotSprings_series = Series(Fayetteville_df_transposed['Average'], dtype='int32')

print('Hot Springs')
print(HotSprings_series.describe())

#Plot the series
from matplotlib import pyplot
HotSprings_series.plot()
pyplot.title('Time Series Plot for House Prices in Fayetteville , Arkansas') 
pyplot.show()

#--------------------------------------------------------------
#Time Series Plot for Searcy, Arkansas metro
Searcy_df =  Zillow_df[Zillow_df.Metro == 'Searcy']

#Calc Average
Searcy_df.loc['Average'] =  Searcy_df.mean()

#flip the data
Searcy_df_t = Searcy_df.transpose()
Searcy_df_transposed= Searcy_df_t.iloc[7:]
Searcy_df_transposed= Searcy_df_transposed.drop(Searcy_df_transposed.columns[:4], axis=1)

#Create the series
HotSprings_series = Series(Searcy_df_transposed['Average'], dtype='int32')

print('Hot Springs')
print(HotSprings_series.describe())

#Plot the series
from matplotlib import pyplot
HotSprings_series.plot()
pyplot.title('Time Series Plot for House Prices in Searcy , Arkansas') 
pyplot.show()

#-----------------------------------------------------------------------
#Model Development
#----------------------------------------------------------------------

pyplot.style.use('fivethirtyeight')

# Selecting Comal County, Kaufman County, Walton County, Midland County, Osceola County,
# St Johns County and Hood County only per intial research

Zillow_df_sel = Zillow_df[Zillow_df['CountyName' ].isin(['Comal County','Kaufman County','Walton County',
'Midland County','Osceola County','St. Johns County','Hood County'])]


Zillow_df_sel = Zillow_df_sel[Zillow_df_sel['State' ].isin(['TX','FL'])]


#Flip the dataframe since it is a timeseries
Zillow_transposed = Zillow_df_sel.transpose()

#find how many zipcodes are there
Num = len(Zillow_transposed.columns)


i=0
while (i < Num):
    
  print("Value of x is ",i)
  #create the dataframe
  Zipcode_1 = pd.DataFrame(Zillow_transposed.iloc[:,i])
  
  #print(Zipcode_1.index.str[0:4])
  Zipcode_1 = Zipcode_1[Zipcode_1.index.str[0:4]!='1996']
  Zipcode_1 = Zipcode_1[Zipcode_1.index.str[0:4]!='2018']
  Zipcode_1 = Zipcode_1[Zipcode_1.index.str[0:4]!='2019']
  Zipcode_df = Zipcode_1.iloc[7:,:]


  Zipcode_df['Date'] = Zipcode_df.index
  Zipcode_df = Zipcode_df.rename(columns = {Zipcode_df.columns.values[0]:'Price'})

  # Rename columns for prophet
  Zipcode_df  = Zipcode_df .rename(index=str, columns={"Price": "y", "Date": "ds"})

  ax = Zipcode_df.set_index('ds').plot(figsize=(12, 8))
  ax.set_ylabel('Monthly Price')
  ax.set_xlabel('Date')

  #pyplot.show()

  # MODEL
  # Set the uncertainty interval to 95% (the Prophet default is 80%)
  price_model = Prophet(interval_width=0.95)
  price_model.fit(Zipcode_df)

  future_dates = price_model.make_future_dataframe(periods=13, freq='M')
  future_dates.tail(15)

  forecast = price_model.predict(future_dates)
  print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(13))

  
  if (i == 0 ):
    print('\n\n\ntesting')
    #Create a dataframe to store the values
    Pred_Price_2018_df = Zipcode_1.iloc[:6,:]
    Pred_Price_2018_df['Column'] = Pred_Price_2018_df.index
    Pred_Price_2018_df = Pred_Price_2018_df.rename(columns = {Zipcode_1.columns.values[0]:'Data'})

    #swap the columns
    Pred_Price_2018_df = Pred_Price_2018_df[['Column', 'Data']]
    Pred_Price_2018_df.reset_index()

    #Add the predicted price for 2018
    d = pd.DataFrame(forecast[['ds','yhat']].tail(12))
    d = d.rename(columns={'ds':'Column','yhat':'Data'})
    Pred_Price_2018_df = Pred_Price_2018_df.append(d)

    result = Pred_Price_2018_df.reset_index() 
    Pred_Price_2018_df= result.iloc[:, 1:3]
    Pred_Price_2018_df = Pred_Price_2018_df.rename(columns = {'Data':'Zipcode_0'})
  else : 
      print('\n\n\ntest')
      data_df = Zipcode_1.iloc[:6,:]
      
      #Add the predicted price for 2018
      d = pd.DataFrame(forecast[['yhat']].tail(12))
      d=d.rename(columns={'yhat':Zipcode_1.columns.values[0]})
      
      data_df = data_df.append(d)
      
      # Append this to Pred_Price_2018_df
      #Pred_Price_2018_df = pd.concat([Pred_Price_2018_df, data_df], ignore_index = True)
      #Pred_Price_2018_df.join(data_df.rename(columns={0:'x'}))
      colname = 'Zipcode_' + str(i)
      Pred_Price_2018_df = Pred_Price_2018_df.reset_index(drop=True)
      data_df= data_df.reset_index(drop=True)
      Pred_Price_2018_df[colname] = data_df
   
      
  i=i+1
  
#Plot the model results
price_model.plot(forecast, uncertainty=True)

#Calc the average price predicted for 2018
data_2018 = Pred_Price_2018_df.iloc[6:,1:]
data_2018.loc['Average_2018'] =  data_2018.mean()

#Calc the average price for 2017 from the data available
data_2017= Zillow_transposed[Zillow_transposed.index.str[0:4]=='2017']
data_2017.loc['Average_2017']  = data_2017.mean()



#Calc the % in price difference between 2018 and 2017
Num = len(data_2017.columns)

data_2017.loc['Average_2018'] = 0

i=0
while (i < Num):
   data_2017.iloc[-1,i] = data_2018.iloc[-1,i]
   i=i+1
    
data_2017.loc['Diff_2017_2018'] = 0
  
i=0
while (i < Num):
   data_2017.iloc[-1,i] = data_2017.iloc[-2,i] - data_2017.iloc[-3,i]
   i=i+1
   

data_2017.loc['Per_Diff_2017_2018'] = 0 
i=0
while (i < Num):
   data_2017.iloc[-1,i] = (data_2017.iloc[-2,i] * 100)/data_2017.iloc[-4,i]
   i=i+1
      
   
##Find the top 3 zipcodes
Per_Diff_2017_2018_df = pd.to_numeric(data_2017.loc['Per_Diff_2017_2018'])
Per_Diff_2017_2018_sorted = Per_Diff_2017_2018_df.sort_values()
top_3_zipcodes = pd.DataFrame(Per_Diff_2017_2018_sorted.tail(3))
#print(list(top_3_zipcodes.index) )
#print(Zillow_df[Zillow_df.index.isin(list(top_3_zipcodes.index))])

top_3_zipcodes_data = Zillow_df[Zillow_df.index.isin(list(top_3_zipcodes.index))]
top_3_zipcodes_data['Per_Diff_2017_2018'] = top_3_zipcodes['Per_Diff_2017_2018']

#Move last column to first
cols = list(top_3_zipcodes_data.columns)
cols = [cols[-1]] + cols[:-1]
top_3_zipcodes_data = top_3_zipcodes_data[cols]

#Visualizing the result

V_df = Pred_Price_2018_df.iloc[7:,1:]
V_df.plot()