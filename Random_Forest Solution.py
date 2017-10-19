"""
Hack Night++ 

A major Coffee distributing company from Colombia is experiencing problems with 
the collection of the grains due to workers strike. 

They are concern with changes in the orders of coffee from their major customers in Colombia 
and other South American countries. 

The CEO of the company, wants to predict how much coffee his customers will order 
once the Price_lb_COL of the pound of coffee reaches COL$ 9000. 

DATA:
    Customer - Unique ID for each customer, 3 or 4 digits number. In 3 digit numbers, the 
    first digit is the country. In the 4 digit, the first two digit represent the country. 
        1 Colombia
        2 Ecuador
        3 Brazil
        4 Argentina
        5 Chile
        6 Venezuela
        7 Paraguay
        8 Bolivia
        9 Uruguay
        10 Peru
        
    Date - Date in which the order was placed
    
    Price_lb_COl - Price in colombian pesos of a pound of Coffee the day of the purchase
    
    Amount_lbs - Total amount of pounds sold in each order

QUESTION: Can you built a model that predicts the demand of coffee for each customer when 
the value of the pound of coffee reaches COL$ 9000 
	
This example is made on
Python 3.6.1 :: Anaconda 4.4.0 (x86_64) using Spyder 3.2.2
All modules with latest version as of Sep 2017

RESPONSE 1: USING RANDOM FOREST:
########################################
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_sql_query as rsq
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score

# Custom functions
def date_std (df):  
    # Transform data to delta-date
    df['Date'] = pd.to_datetime(df['Date']) 
    min_dt= min(df['Date'])
    st_df = pd.DataFrame()
    for Customer in df.Customer.unique():
        df1 = df[df['Customer'] == Customer]
        df1['date_delta'] = (df1['Date'] - min_dt) / np.timedelta64(1,'D')
        st_df = pd.concat([st_df, df1])
    st_df = st_df.drop(['Date'],axis=1)
    return st_df

def run_model (model, train, test, predictors, predicted): 
    # Runs the specified model and calculates MSE and obb score
    model.fit(train[predictors],train[predicted])
    predictions = model.predict(test[predictors])
    MSE1 = mean_squared_error(predictions,test[predicted].values)
    MSE = "{0:6.2f}".format(MSE1)
    obb_score = "{0:.4f}".format(model.oob_score_)
    return float(MSE), float(obb_score)

def demand_prediction(customer, Price_lb_COL, model):
    # Predictd demand for a single customer
    df_demand=pd.DataFrame([[customer,Price_lb_COL]],columns = n_predictors)
    demand = model.predict(df_demand)
    return round(demand[0])
    
def pred_per_customer(customer_list, Price_lb_COL, model):
    # Generates a DF with predicion for all customers on a list
    df=pd.DataFrame()
    for customer in range(len(customer_list)):
        customer = customer_list[customer]
        pred = demand_prediction(customer,Price_lb_COL, model)
        df2 = pd.DataFrame([[customer,Price_lb_COL, pred]], columns=['Customer','Price_lb_COL','Pred_DEMAND']) 
        df = df.append(df2, ignore_index=True)
    return df


# READ DATA
# From file
product_sales = pd.read_csv('/Users/jpinzon/Google Drive/01_GitHub/Hack_meetup/Hack_01/ML_hack_night/data.csv')
product_sales = product_sales.sort_values('Customer').reset_index(drop=True)

# Data exploration
product_sales.head(10)
product_sales.describe()

# Visualization
product_sales['Price_lb_COL'].hist(bins=50)
product_sales['Amount_lbs'].hist(bins=50)

product_sales.boxplot(column='Price_lb_COL')
product_sales.boxplot(column='Amount_lbs')


product_sales.boxplot(column='Price_lb_COL', by = 'Customer')
product_sales.boxplot(column='Amount_lbs', by = 'Customer')

# Seems some demand values are larger than the rest 
for group in product_sales['Customer'].unique():
    df = product_sales[product_sales['Customer']==group]
    plt.scatter(df['Price_lb_COL'], df['Amount_lbs'])

# Remove those values
product_sales['Date'] = pd.to_datetime(product_sales['Date']) 
product_sales = product_sales.set_index(['Date', 'Customer'])
product_sales = product_sales[product_sales.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]
product_sales = product_sales.reset_index()

# Confirm
for group in product_sales['Customer'].unique():
    df = product_sales[product_sales['Customer']==group]
    plt.scatter(df['Price_lb_COL'], df['Amount_lbs'])
    
product_sales.boxplot(column='Amount_lbs', by = 'Customer')

# Format Date as delta time for each group
product_sales = date_std(product_sales)

# Encoding numeric columns as int64    
product_sales.dtypes


le = LabelEncoder()
for i in ('date_delta', 'Price_lb_COL', 'Amount_lbs'):
    product_sales[i] = le.fit_transform(product_sales[i])
    
product_sales.dtypes

product_sales['Customer'] = product_sales['Customer'].astype(str)




# Split the data into train and test datasets
df_train, df_test = train_test_split(product_sales, test_size = 0.25)

# RANDOM FOREST REGRESSION
# Define features
predictors = ['Customer','date_delta','Price_lb_COL']
outcome = 'Amount_lbs'
"""
CAN JUMP TO THE OPTIMAZED MODEL BELOW
"""
# Raw Model
rf_reg_model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=1e-07, min_samples_leaf=1,
           min_samples_split=2, min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=True, random_state=None,
           verbose=0, warm_start=False)

scoring = run_model(rf_reg_model, df_train, df_test, predictors, outcome)
scoring

# MODEL OPTIOMIZATION
# Criteria
criteria=['mae', 'mse']
for crit in criteria:
    rf_reg_model = RandomForestRegressor(bootstrap=True, criterion=crit, 
                                   max_depth=None,max_features='auto', 
                                   max_leaf_nodes=None, min_impurity_decrease=1e-07,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, n_estimators=100, 
                                   n_jobs=1, oob_score=True, random_state=None,
                                   verbose=0, warm_start=False)
    scoring = run_model(rf_reg_model, df_train, df_test, predictors, outcome)
    print (crit, '\n', scoring[0], scoring[1])
 
# Features
features=[0.3,0.5,1,2,3,'auto']
for feat in features:
    rf_reg_model = RandomForestRegressor(bootstrap=True, criterion='mse', 
                                   max_depth=None,max_features=feat, 
                                   max_leaf_nodes=None, min_impurity_decrease=1e-07,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, n_estimators=100, 
                                   n_jobs=1, oob_score=True, random_state=None,
                                   verbose=0, warm_start=False)
    scoring = run_model(rf_reg_model, df_train, df_test, predictors, outcome)
    print (feat, '\n', scoring[0], scoring[1])
    
# n_estimators
estimators=[100, 500, 1000, 2000]
for est in estimators:
    rf_reg_model = RandomForestRegressor(bootstrap=True, criterion='mse', 
                                   max_depth=None,max_features='auto', 
                                   max_leaf_nodes=None, min_impurity_decrease=1e-07,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, n_estimators=est, 
                                   n_jobs=1, oob_score=True, random_state=None,
                                   verbose=0, warm_start=False)
    scoring = run_model(rf_reg_model, df_train, df_test, predictors, outcome)
    print (est, '\n', scoring[0], scoring[1])

# n_jobs
jobs=[1,2,3, 4, 5]
for job in jobs:
    rf_reg_model = RandomForestRegressor(bootstrap=True, criterion='mse', 
                                   max_depth=None,max_features='auto', 
                                   max_leaf_nodes=None, min_impurity_decrease=1e-07,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, n_estimators=2000, 
                                   n_jobs=job, oob_score=True, random_state=None,
                                   verbose=0, warm_start=False)
    scoring = run_model(rf_reg_model, df_train, df_test, predictors, outcome)
    print (job, '\n', scoring[0], scoring[1])

 
"""
OPTIMIZED MODEL
"""
# TRAINING OPTIMIZED MODEL
rf_reg_model_opt = RandomForestRegressor(criterion = 'mse', max_features = 'auto',
                                   n_estimators = 2000,  n_jobs = 3, 
                                   oob_score = True, verbose = 0)
scoring = run_model(rf_reg_model_opt, df_train, df_test, predictors, outcome)
scoring

feature_imp_opt = pd.DataFrame(rf_reg_model_opt.feature_importances_, 
                           index=predictors).sort_values(by=[0],ascending=False)
feature_imp_opt

# CROSSVALATION
scores = cross_val_score(rf_reg_model_opt, df_train[predictors], df_train[outcome], cv=5)
mean_score = '{0:0.4f}'.format(scores.mean() )
mean_score

# PREDICTION MODEL 
# Removed Date as it only accounts ~5%
rf_reg_model_opt_pred = rf_reg_model_opt
n_predictors=['Customer', 'Price_lb_COL']
scoring_pred = run_model(rf_reg_model_opt_pred, df_train, df_test, n_predictors, outcome)

feature_imp_pred = pd.DataFrame(rf_reg_model_opt_pred.feature_importances_, 
                                index=n_predictors).sort_values(by=[0],ascending=False)
feature_imp_pred

# PREDICTING DEMAND
all_cust = list(product_sales['Customer'].unique())
# Prediction at U$3. Can be changed to any value 
pred_per_customer(all_cust, 3, rf_reg_model_opt_pred)

