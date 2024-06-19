import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso

# Ignore warnings
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

# Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/rushilcs/Time-Series-Forcasting/main/time%20series%20data%20-%20RDC_Inventory_Core_Metrics_County_History.csv")

# Function to get location abbreviation
def getLocation(location, type_location):
    states = {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
        "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
        "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
        "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
        "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
        "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
        "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
        "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
        "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
        "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming"
    }
    if type_location == 'state':
        for ab in states:
            if states[ab].lower() == location.lower():
                return ab.lower()
    if type_location == 'county':
        flag = False
        while not flag:
            for ab in states:
                if states[ab].lower() == location.lower():
                    print(location + "'s abbreviation is " + ab.lower())
                    location = input("Please enter a county in the United States in the following format: county, state abbreviation. \nIf needed, enter a state to see its abbreviation: ")
            flag = True
        return location.lower()

# Translate dates from yyyymm to usable dates for the model
def translate_time(yyyymm):
    year_string = str(yyyymm)
    year = int(year_string[:4])
    month = int(year_string[4:])
    float_year = year + month / 12.0
    return float_year

# Get date from user and translate it into a usable value
def getDate():
    val = input("Please enter a future date for the prediction in the format yyyymm: ")
    val1 = int(val)
    flag = False
    while not flag:
        if val1 < 202207:
            val = input("Invalid Date. Please enter a future date for the prediction in the format yyyymm: ")
            val1 = int(val)
        else:
            flag = True
    return translate_time(val1)

# Process a new dataframe based on location type
def process_location(location):
    if len(location) > 2:
        df = data[data['county_name'] == location].reset_index(drop=True)
    else:
        df = data[data['county_name'].str[-2:].str.lower() == location].reset_index(drop=True)
    df['month_date_yyyymm'] = df['month_date_yyyymm'].apply(translate_time)
    return df

# Train and test the model
def train_test(df, date):
    x1 = np.array(df['month_date_yyyymm']).reshape(-1, 1)
    y1 = np.array(df['median_listing_price_per_square_foot']).reshape(-1, 1)
    regressor = LinearRegression()
    regressor.fit(x1, y1)
    median_listing_price_per_square_foot = regressor.predict([[date]])

    X = df[['month_date_yyyymm', 'median_days_on_market', 'median_listing_price_per_square_foot']].values
    y = df['median_listing_price'].values
    model_l = Lasso(alpha=1)
    model_l.fit(X, y)
    predicted_price = model_l.predict([[date, df['median_days_on_market'].iloc[0], median_listing_price_per_square_foot[0][0]]])
    
    print('Current price: ', df['median_listing_price'].iloc[0])
    print('Predicted price: ', predicted_price[0])

# General function to run the program
if __name__ == "__main__":
    print("---Predict a House's Cost in the Future!---\n")
    running = True
    while running:
        type_location = input("Would you like to predict a certain county or an entire state? Enter either 'county' or 'state' to specify your query: ")
        if type_location == 'state':
            print("Note that this method's Linear Regression is very inaccurate!")
            location = input("Enter the full name of the state you wish to look at: ")
            loc = getLocation(location, 'state')
        elif type_location == 'county':
            location = input("Please enter a county in the United States in the following format: county, state abbreviation. \nIf needed, enter a state to see its abbreviation: ")
            loc = getLocation(location, 'county')
        else:
            print("Not a valid input, try again.")
            continue
        
        date = getDate()
        df1 = process_location(loc)
        train_test(df1, date)
        
        retry = input("Query another location/year? (Y/N): ")
        if retry.lower() == "y":
            continue
        elif retry.lower() == "n":
            running = False
        else:
            print("Not a valid input. Please try again.")
