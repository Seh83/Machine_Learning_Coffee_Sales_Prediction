# Predict Coffee Demand in South America

# About

This Git repo is sponsored by the [Hack My Life] (https://meetup.com/hack-my-life/) meetup group of Richardson, TX. Special thanks also goes out the slack community of [RemoteCoder.net] (http://www.remotecoder.net) for their support.

# Exercise
Data has been adapted for the purpose of the exercise. 

# Description
A major Coffee distributing company from Colombia is experiencing problems with 
the collection of the grains due to workers strike. 

They are concern with changes in the orders of coffee from their major customers in Colombia 
and other South American countries. 

The CEO of the company, wants to predict how much coffee his customers will order 
once the Price_lb_COL of the pound of coffee reaches COL$ 9000. 

# Data
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

# Question
Can you built a model that predicts the demand of coffee for each customer when 
the value of the pound of coffee reaches COL$ 9000 
	
# This example is made on
Python 3.6.1 :: Anaconda 4.4.0 (x86_64) using Spyder 3.2.2

All modules with latest version as of Sep 2017

Pandas
Numpy
Sklearn
Matplotlib

We provide data and code to solve the challenge. 
The code includes the solutions using a Random Forrests model, but other models are possible. We dare you to try other algorithms and provide your feedback in the comments. 

# Usage
Download/clone the git 
Open the Random_Forest Solution.py file in spyder and execute each line at a time
