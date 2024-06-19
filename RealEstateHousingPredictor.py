from pandas.core import flags
import warnings
import pandas as pd

##from pandas.core.common import SettingWithCopyWarning

#warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import numpy as np

import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model

data=pd.read_csv("https://raw.githubusercontent.com/rushilcs/Time-Series-Forcasting/main/time%20series%20data%20-%20RDC_Inventory_Core_Metrics_County_History.csv")

#gets a location from user input to use in processing dataframe: if state, then get state abbrevation OR if county, then get county and state abbreviation
def getLocation(location, type_location):
  if type_location == 'state':
    states = {"AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"}
    for ab in states:
      if states[ab].lower() == location.lower():
        return ab.lower()
  if type_location == 'county':
    val = input("Please enter a county in the United States in the following format: county, state abbreviation. \nIf needed, enter a state to see its abbreviation.")
    states = {"AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"}
    flag = False
    while flag == False:
      for ab in states:
        if states[ab].lower() == val.lower():
          print (val + "'s abbreviation is " + ab.lower())
          val = input("Please enter a county in the United States in the following format: county, state abbreviation. \nIf needed, enter a state to see its abbreviation.")
      flag = True
    return(val.lower())

#translates dates from yyyymm to useable dates for the model
def translate_time(yyyymm):
    year_string = str(yyyymm)
    year = year_string[:4]
    y = float(year)
    month = year_string[4:]
    m = float(month) / 12
    float_year = y + m
    return float_year

#gets date from user and translates it into a useable value
def getDate():
  val = input("Please enter a future date for the prediction in the format yyyymm: ")
  val1 = int(val)
  flag = False
  while flag == False:
    val1 = int(val)
    if val1 < 202207:
      val = input("Invalid Date. Please enter a future date for the prediction in the format yyyymm: ")
    else:
      flag = True
  val2 = translate_time(val1)
  return val2

#processes a new dataframe based on location type
def process_location(location):
  #if location is a county, then new dataframe has rows only pertaining to that county
  if len(location) > 2:
    df = data[(data['county_name'] == location)]
    df = df.reset_index()
    for i in range (0, len(df.index)):
      df.loc[i, 'month_date_yyyymm'] = translate_time(df.loc[i, 'month_date_yyyymm'])
    return df
  #if location is a state, then new dataframe has rows only pertaining to that state
  else:
    df = data[(data['county_name'].str[-2:].str.contains(location))]
    df = df.reset_index()
    for i in range (0, len(df.index)):
      df.loc[i, 'month_date_yyyymm'] = translate_time(df.loc[i, 'month_date_yyyymm'])
    return df

def train_test(df):
    #use linear regression to predict median_listing_price_per_square_foot factor at given date
    x1 = np.array(df['month_date_yyyymm']).reshape(-1, 1)
    y1 = np.array(df['median_listing_price_per_square_foot']).reshape(-1,1)
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(x1, y1)
    median_listing_price_per_square_foot = regressor.predict([[date]])

    #train model
    X = df.values[:, 2:5]  # get input values
    y = df.values[:, 5]  # get output values
    m = len(y) # Number of training examples
    print('Total no of training examples (m) = %s \n' %(m))

    #use lasso linear regression to predict median listing price in county
    model_l = linear_model.Lasso(alpha= 1)
    model_l.fit(X,y)
    print('coef= ' , model_l.coef_)
    print('intercept= ' , model_l.intercept_)
    price = model_l.predict([[date, df.loc[0, 'median_days_on_market'] , median_listing_price_per_square_foot]])
    print('Current price: ', df.loc[0, 'median_listing_price'])
    print('Predicted price: ', price)




  # General Function to run the program
if __name__ == "__main__":
    data=pd.read_csv("https://raw.githubusercontent.com/rushilcs/Time-Series-Forcasting/main/time%20series%20data%20-%20RDC_Inventory_Core_Metrics_County_History.csv")
    running = True
    print("---Predict a House's Cost in the Future!---\n")
    while running:
        type_location = input("Would you like to predict a certain county or an entire state? Enter either 'county' or 'state' to specify your query: ")
        if type_location == 'state':
            print("Note that this method's Linear Regression is very inaccurate!")
            #gets state and appropriate location to use in processing data
            location = input("Enters the full name of the state you wish to look at: ")
            loc = getLocation(location, 'state')
        elif type_location == 'county':
          #gets county and appropriate location to use in processing data
            loc = getLocation(0, 'county')
        else:
            print("Not a valid input, try again.")
            continue
        #gets date, uses previous location to process dataframe, then uses date and processed dataframe to make prediction
        date = getDate()
        df1 = process_location(loc)
        train_test(df1)
        #Asks user if they want to repeat the process for another location/year, and repeats program if 'Y' is entered. Stops program if 'N' is entered.
        retry = input("Query another location/year? (Y/N): ")
        if retry.lower() == "y":
            continue
        elif retry.lower() != "n":
            print("Not a valid input. Please try again.")
        else:
            running = False