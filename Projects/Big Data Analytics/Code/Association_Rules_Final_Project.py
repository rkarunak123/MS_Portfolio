# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 09:59:12 2019
Association Rule Mining on Zillow Housing Data
@author: Rebecca Karunakaran
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

#-----------------------------------------------------------------------------------
#Load the Data
#------------------------------------------------------------------------------------ 

CA_df = pd.read_csv('C:\Syracuse\CAZipPopulations_healthcare_metro.csv')


#--------------------------------------------------------------------------------
# Create Incomebracket, Popbracket categories
#------------------------------------------------------------------------------

#2017
#Home Value Category
def Avg_Price(row):
    val= row[-12:].mean()
    return val

CA_df['Avg_Price_2017'] =CA_df.apply(Avg_Price, axis=1)

Median_Price_2017 = CA_df['Avg_Price_2017'].mean()

Below_Median_Range = Median_Price_2017*2/3
Median_Range = Median_Price_2017*2/3*2

def Price_Cat(row):
    if row['Avg_Price_2017'] < Below_Median_Range :
        val = 'BelowMedianHomePrice'
    elif (row['Avg_Price_2017'] > Below_Median_Range and  row['Avg_Price_2017']  < Median_Range ):
        val = 'MedianHomePrice'
    else:
        val = 'AboveMedianHomePrice'

    return val

CA_df['HomePriceCat_2017'] = CA_df.apply(Price_Cat, axis=1)

#Incomebracket
def Incomebracket(row):
    if row['Total_Income_2017'] < 10000 :
        val = 'LowIncome'
    elif row['Total_Income_2017'] < 20000 :
        val = 'MiddleIncome'
    elif row['Total_Income_2017'] < 30000:
        val = 'UpperMiddleIncome'
    else:
        val = 'UpperIncome'
    return val

CA_df['Incomebracket_2017'] = CA_df.apply(Incomebracket, axis=1)

#Population Category
def PopCat(row):
    if row['Population_2017'] < 2500:
        val = 'Rural'
    elif row['Population_2017'] <50000 :
        val = 'Semi-Urban'
    else:
        val = 'Urban'
    return val

CA_df['PopDenbracket_2017'] = CA_df.apply(PopCat, axis=1)

#Native or Foreign born
def Nat_For(row):
    if row['Population_2017']/2 < row['Foreign_Born_2017']:
        val = 'ForeignBorn'
    else:
        val = 'NativeBorn'
    return val

CA_df['PopType_2017'] = CA_df.apply(Nat_For, axis=1)

#Ethnicity
Pop_Ethnic_df = CA_df[["Population_White_2017", "Population_Black_2017", "Population_Asian_2017","Population_Other_2017"]]


CA_df['Ethnicity2017'] = Pop_Ethnic_df.idxmax(axis = 1)
CA_df['Ethnicity_2017'] = CA_df.Ethnicity2017.str[11:16]

#Housing Type
def HousingType(row):
    if row['Home_Ownership_2017'] > row['Renter_2017']:
        val = 'Owners'
    else:
        val = 'Renters'
    return val

CA_df['HousingType_2017'] = CA_df.apply(HousingType, axis=1)

#Below Poverty
BelowPoverty_df = CA_df[["Poverty_White_2017", "Poverty_Black_2017", "Poverty_Hispanic_2017"]]


CA_df['Poverty2017'] = BelowPoverty_df.idxmax(axis = 1)
CA_df['BelowPoverty_2017'] = 'Below_Poverty_' + CA_df.Poverty2017.str[8:13]

#2016

#Home Value Category
def Avg_Price(row):
    val= row[-24:-13].mean()
    return val

CA_df['Avg_Price_2016'] =CA_df.apply(Avg_Price, axis=1)

Median_Price_2017 = CA_df['Avg_Price_2016'].mean()

Below_Median_Range = Median_Price_2017*2/3
Median_Range = Median_Price_2017*2/3*2

def Price_Cat(row):
    if row['Avg_Price_2016'] < Below_Median_Range :
        val = 'BelowMedianHomePrice'
    elif (row['Avg_Price_2017'] > Below_Median_Range and  row['Avg_Price_2017']  < Median_Range ):
        val = 'MedianHomePrice'
    else:
        val = 'AboveMedianHomePrice'

    return val

CA_df['HomePriceCat_2016'] = CA_df.apply(Price_Cat, axis=1)

#Incomebracket
def Incomebracket(row):
    if row['Total_Income_2016'] < 10000 :
        val = 'LowIncome'
    elif row['Total_Income_2016'] < 20000 :
        val = 'MiddleIncome'
    elif row['Total_Income_2016'] < 30000:
        val = 'UpperMiddleIncome'
    else:
        val = 'UpperIncome'
    return val

CA_df['Incomebracket_2016'] = CA_df.apply(Incomebracket, axis=1)

#Population Category
def PopCat(row):
    if row['Population_2016'] < 2500:
        val = 'Rural'
    elif row['Population_2016'] <50000 :
        val = 'Semi-Urban'
    else:
        val = 'Urban'
    return val

CA_df['PopDenbracket_2016'] = CA_df.apply(PopCat, axis=1)

#Native or Foreign born
def Nat_For(row):
    if row['Population_2016']/2 < row['Foreign_Born_2016']:
        val = 'ForeignBorn'
    else:
        val = 'NativeBorn'
    return val

CA_df['PopType_2016'] = CA_df.apply(Nat_For, axis=1)

#Ethnicity
Pop_Ethnic_df = CA_df[["Population_White_2016", "Population_Black_2016", "Population_Asian_2016","Population_Other_2016"]]


CA_df['Ethnicity2016'] = Pop_Ethnic_df.idxmax(axis = 1)
CA_df['Ethnicity_2016'] = CA_df.Ethnicity2016.str[11:16]

#Housing Type
def HousingType(row):
    if row['Home_Ownership_2016'] > row['Renter_2016']:
        val = 'Owners'
    else:
        val = 'Renters'
    return val

CA_df['HousingType_2016'] = CA_df.apply(HousingType, axis=1)

#Below Poverty
BelowPoverty_df = CA_df[["Poverty_White_2016", "Poverty_Black_2016", "Poverty_Hispanic_2016"]]


CA_df['Poverty2016'] = BelowPoverty_df.idxmax(axis = 1)
CA_df['BelowPoverty_2016'] = CA_df.Poverty2016.str[8:13]

#2015

#Home Value Category
def Avg_Price(row):
    val= row[-36:-25].mean()
    return val

CA_df['Avg_Price_2015'] =CA_df.apply(Avg_Price, axis=1)

Median_Price_2017 = CA_df['Avg_Price_2015'].mean()

Below_Median_Range = Median_Price_2017*2/3
Median_Range = Median_Price_2017*2/3*2

def Price_Cat(row):
    if row['Avg_Price_2015'] < Below_Median_Range :
        val = 'BelowMedianHomePrice'
    elif (row['Avg_Price_2017'] > Below_Median_Range and  row['Avg_Price_2017']  < Median_Range ):
        val = 'MedianHomePrice'
    else:
        val = 'AboveMedianHomePrice'

    return val

CA_df['HomePriceCat_2015'] = CA_df.apply(Price_Cat, axis=1)

#Incomebracket
def Incomebracket(row):
    if row['Total_Income_2015'] < 10000 :
        val = 'LowIncome'
    elif row['Total_Income_2015'] < 20000 :
        val = 'MiddleIncome'
    elif row['Total_Income_2015'] < 30000:
        val = 'UpperMiddleIncome'
    else:
        val = 'UpperIncome'
    return val

CA_df['Incomebracket_2015'] = CA_df.apply(Incomebracket, axis=1)

#Population Category
def PopCat(row):
    if row['Population_2015'] < 2500:
        val = 'Rural'
    elif row['Population_2015'] <50000 :
        val = 'Semi-Urban'
    else:
        val = 'Urban'
    return val

CA_df['PopDenbracket_2015'] = CA_df.apply(PopCat, axis=1)

#Native or Foreign born
def Nat_For(row):
    if row['Population_2015']/2 < row['Foreign_Born_2015']:
        val = 'ForeignBorn'
    else:
        val = 'NativeBorn'
    return val

CA_df['PopType_2015'] = CA_df.apply(Nat_For, axis=1)

#Ethnicity
Pop_Ethnic_df = CA_df[["Population_White_2015", "Population_Black_2015", "Population_Asian_2015","Population_Other_2015"]]


CA_df['Ethnicity2015'] = Pop_Ethnic_df.idxmax(axis = 1)
CA_df['Ethnicity_2015'] = CA_df.Ethnicity2015.str[11:16]

#Housing Type
def HousingType(row):
    if row['Home_Ownership_2015'] > row['Renter_2015']:
        val = 'Owners'
    else:
        val = 'Renters'
    return val

CA_df['HousingType_2015'] = CA_df.apply(HousingType, axis=1)

#Below Poverty
BelowPoverty_df = CA_df[["Poverty_White_2015", "Poverty_Black_2015", "Poverty_Hispanic_2015"]]


CA_df['Poverty2015'] = BelowPoverty_df.idxmax(axis = 1)
CA_df['BelowPoverty_2015'] = CA_df.Poverty2015.str[8:13]

#2014

#Home Value Category
def Avg_Price(row):
    val= row[-48:-37].mean()
    return val

CA_df['Avg_Price_2014'] =CA_df.apply(Avg_Price, axis=1)

Median_Price_2017 = CA_df['Avg_Price_2014'].mean()

Below_Median_Range = Median_Price_2017*2/3
Median_Range = Median_Price_2017*2/3*2

def Price_Cat(row):
    if row['Avg_Price_2014'] < Below_Median_Range :
        val = 'BelowMedianHomePrice'
    elif (row['Avg_Price_2017'] > Below_Median_Range and  row['Avg_Price_2017']  < Median_Range ):
        val = 'MedianHomePrice'
    else:
        val = 'AboveMedianHomePrice'

    return val

CA_df['HomePriceCat_2014'] = CA_df.apply(Price_Cat, axis=1)

#Incomebracket
def Incomebracket(row):
    if row['Total_Income_2014'] < 10000 :
        val = 'LowIncome'
    elif row['Total_Income_2014'] < 20000 :
        val = 'MiddleIncome'
    elif row['Total_Income_2014'] < 30000:
        val = 'UpperMiddleIncome'
    else:
        val = 'UpperIncome'
    return val

CA_df['Incomebracket_2014'] = CA_df.apply(Incomebracket, axis=1)

#Population Category
def PopCat(row):
    if row['Population_2014'] < 2500:
        val = 'Rural'
    elif row['Population_2014'] <50000 :
        val = 'Semi-Urban'
    else:
        val = 'Urban'
    return val

CA_df['PopDenbracket_2014'] = CA_df.apply(PopCat, axis=1)

#Native or Foreign born
def Nat_For(row):
    if row['Population_2014']/2 < row['Foreign_Born_2014']:
        val = 'ForeignBorn'
    else:
        val = 'NativeBorn'
    return val

CA_df['PopType_2014'] = CA_df.apply(Nat_For, axis=1)

#Ethnicity
Pop_Ethnic_df = CA_df[["Population_White_2014", "Population_Black_2014", "Population_Asian_2014","Population_Other_2014"]]


CA_df['Ethnicity2014'] = Pop_Ethnic_df.idxmax(axis = 1)
CA_df['Ethnicity_2014'] = CA_df.Ethnicity2014.str[11:16]

#Housing Type
def HousingType(row):
    if row['Home_Ownership_2014'] > row['Renter_2014']:
        val = 'Owners'
    else:
        val = 'Renters'
    return val

CA_df['HousingType_2014'] = CA_df.apply(HousingType, axis=1)

#Below Poverty
BelowPoverty_df = CA_df[["Poverty_White_2014", "Poverty_Black_2014", "Poverty_Hispanic_2014"]]


CA_df['Poverty2014'] = BelowPoverty_df.idxmax(axis = 1)
CA_df['BelowPoverty_2014'] = CA_df.Poverty2014.str[8:13]

#2013

#Home Value Category
def Avg_Price(row):
    val= row[-60:-49].mean()
    return val

CA_df['Avg_Price_2013'] =CA_df.apply(Avg_Price, axis=1)

Median_Price_2017 = CA_df['Avg_Price_2013'].mean()

Below_Median_Range = Median_Price_2017*2/3
Median_Range = Median_Price_2017*2/3*2

def Price_Cat(row):
    if row['Avg_Price_2013'] < Below_Median_Range :
        val = 'BelowMedianHomePrice'
    elif (row['Avg_Price_2017'] > Below_Median_Range and  row['Avg_Price_2017']  < Median_Range ):
        val = 'MedianHomePrice'
    else:
        val = 'AboveMedianHomePrice'

    return val

CA_df['HomePriceCat_2013'] = CA_df.apply(Price_Cat, axis=1)

#Incomebracket
def Incomebracket(row):
    if row['Total_Income_2013'] < 10000 :
        val = 'LowIncome'
    elif row['Total_Income_2013'] < 20000 :
        val = 'MiddleIncome'
    elif row['Total_Income_2013'] < 30000:
        val = 'UpperMiddleIncome'
    else:
        val = 'UpperIncome'
    return val

CA_df['Incomebracket_2013'] = CA_df.apply(Incomebracket, axis=1)

#Population Category
def PopCat(row):
    if row['Population_2013'] < 2500:
        val = 'Rural'
    elif row['Population_2013'] <50000 :
        val = 'Semi-Urban'
    else:
        val = 'Urban'
    return val

CA_df['PopDenbracket_2013'] = CA_df.apply(PopCat, axis=1)

#Native or Foreign born
def Nat_For(row):
    if row['Population_2013']/2 < row['Foreign_Born_2013']:
        val = 'ForeignBorn'
    else:
        val = 'NativeBorn'
    return val

CA_df['PopType_2013'] = CA_df.apply(Nat_For, axis=1)

#Ethnicity
Pop_Ethnic_df = CA_df[["Population_White_2013", "Population_Black_2013", "Population_Asian_2013","Population_Other_2013"]]


CA_df['Ethnicity2013'] = Pop_Ethnic_df.idxmax(axis = 1)
CA_df['Ethnicity_2013'] = CA_df.Ethnicity2013.str[11:16]

#Housing Type
def HousingType(row):
    if row['Home_Ownership_2013'] > row['Renter_2013']:
        val = 'Owners'
    else:
        val = 'Renters'
    return val

CA_df['HousingType_2013'] = CA_df.apply(HousingType, axis=1)

#Below Poverty
BelowPoverty_df = CA_df[["Poverty_White_2013", "Poverty_Black_2013", "Poverty_Hispanic_2013"]]


CA_df['Poverty2013'] = BelowPoverty_df.idxmax(axis = 1)
CA_df['BelowPoverty_2013'] = CA_df.Poverty2013.str[8:13]

#2012

#Home Value Category
def Avg_Price(row):
    val= row[-72:-61].mean()
    return val

CA_df['Avg_Price_2012'] =CA_df.apply(Avg_Price, axis=1)

Median_Price_2017 = CA_df['Avg_Price_2012'].mean()

Below_Median_Range = Median_Price_2017*2/3
Median_Range = Median_Price_2017*2/3*2

def Price_Cat(row):
    if row['Avg_Price_2012'] < Below_Median_Range :
        val = 'BelowMedianHomePrice'
    elif (row['Avg_Price_2017'] > Below_Median_Range and  row['Avg_Price_2017']  < Median_Range ):
        val = 'MedianHomePrice'
    else:
        val = 'AboveMedianHomePrice'

    return val

CA_df['HomePriceCat_2012'] = CA_df.apply(Price_Cat, axis=1)

#Incomebracket
def Incomebracket(row):
    if row['Total_Income_2012'] < 10000 :
        val = 'LowIncome'
    elif row['Total_Income_2012'] < 20000 :
        val = 'MiddleIncome'
    elif row['Total_Income_2012'] < 30000:
        val = 'UpperMiddleIncome'
    else:
        val = 'UpperIncome'
    return val

CA_df['Incomebracket_2012'] = CA_df.apply(Incomebracket, axis=1)

#Population Category
def PopCat(row):
    if row['Population_2012'] < 2500:
        val = 'Rural'
    elif row['Population_2012'] <50000 :
        val = 'Semi-Urban'
    else:
        val = 'Urban'
    return val

CA_df['PopDenbracket_2012'] = CA_df.apply(PopCat, axis=1)

#Native or Foreign born
def Nat_For(row):
    if row['Population_2012']/2 < row['Foreign_Born_2012']:
        val = 'ForeignBorn'
    else:
        val = 'NativeBorn'
    return val

CA_df['PopType_2012'] = CA_df.apply(Nat_For, axis=1)

#Ethnicity
Pop_Ethnic_df = CA_df[["Population_White_2012", "Population_Black_2012", "Population_Asian_2012","Population_Other_2012"]]


CA_df['Ethnicity2012'] = Pop_Ethnic_df.idxmax(axis = 1)
CA_df['Ethnicity_2012'] = CA_df.Ethnicity2012.str[11:16]

#Housing Type
def HousingType(row):
    if row['Home_Ownership_2012'] > row['Renter_2012']:
        val = 'Owners'
    else:
        val = 'Renters'
    return val

CA_df['HousingType_2012'] = CA_df.apply(HousingType, axis=1)

#Below Poverty
BelowPoverty_df = CA_df[["Poverty_White_2012", "Poverty_Black_2012", "Poverty_Hispanic_2012"]]


CA_df['Poverty2012'] = BelowPoverty_df.idxmax(axis = 1)
CA_df['BelowPoverty_2012'] = CA_df.Poverty2012.str[8:13]




#Export to Excel
#CA_df.to_csv(r'C:\Syracuse\CA_Zip_ForJosh.csv',index = None, header=True) 

#--------------------------------------------------------------------------------
# Association Rule Mining
#------------------------------------------------------------------------------
CAData = CA_df[['HomePriceCat_2017', 'Incomebracket_2017','PopDenbracket_2017','PopType_2017','Ethnicity_2017','HousingType_2017','BelowPoverty_2017']]
#CAData['Zipcode'] = CAData['Zipcode'].astype(str)

CAData_list = CAData.values.tolist()

te = TransactionEncoder()
te_ary = te.fit(CAData_list).transform(CAData_list)
df = pd.DataFrame(te_ary, columns=te.columns_)

#Bird's Eye View - Support at 0.1

frequent_itemsets = apriori(df, min_support=0.1,use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets[ (frequent_itemsets['length'] >= 2) ])

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
#print(rules)


rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))

rules_final_highlevel = rules[ (rules['antecedent_len'] >= 4) &
       (rules['confidence'] >= 0.9) &
       (rules['lift'] > 1.0)  ]



#Bringing support down to .01

frequent_itemsets = apriori(df, min_support=0.01,use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets[ (frequent_itemsets['length'] >= 2) ])

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
#print(rules)


rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))

rules_final_drilldown = rules[ (rules['antecedent_len'] >= 5) &
       (rules['confidence'] >= 0.8) &
       (rules['lift'] > 1.0)  ]


##Bringing support down to .005 

frequent_itemsets = apriori(df, min_support=0.005,use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets[ (frequent_itemsets['length'] >= 2) ])

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
#print(rules)


rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))

rules_final_evidence_gentri = rules[ (rules['antecedent_len'] >= 5) &
       (rules['confidence'] >= 0.9) &
       (rules['lift'] > 1.0)  ]


##Export to Excel
##rules_final.to_csv(r'C:\Syracuse\Rules_ForJosh.csv',index = None, header=True) 
