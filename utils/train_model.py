"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')


 #Drop the Unnamed:0 column
train = train.drop(['Unnamed: 0'], axis=1)

#Replace null values in Valencia_pressure with Madrid_pressure
train.loc[train['Valencia_pressure'].isna(),'Valencia_pressure'] = \
train.loc[train['Valencia_pressure'].isna(), 'Madrid_pressure']

#Convert time to numpy datetime object
train['time'] = pd.to_datetime(train['time'], format='%Y-%m-%d %H:%M:%S')

#Split time column to hour, day, month & year
train['year'] = pd.DatetimeIndex(train['time']).year
train['month'] = pd.DatetimeIndex(train['time']).month
train['day'] = pd.DatetimeIndex(train['time']).day
train['hour'] = pd.DatetimeIndex(train['time']).hour

#Drop the time column
train = train.drop(columns= 'time')

# Re-organize the column features to have date features at the start
col_titles = ['year'] + ['month'] + ['day'] + ['hour'] + \
    [col for col in train.columns \
    if col not in ['year', 'month', 'day', 'hour','load_shortfall_3h']] + \
        ['load_shortfall_3h']

#Create dummy variables (winter, summer, autumn, spring) based on
# weather seasons using the month column
train.loc[train['month'].isin([1,2,3]),['winter','spring','summer','autumn']] = [1,0,0,0]
train.loc[train['month'].isin([4,5,6]),['winter','spring','summer','autumn']] = [0,1,0,0]
train.loc[train['month'].isin([7,8,9]),['winter','spring','summer','autumn']] = [0,0,1,0]
train.loc[train['month'].isin([10,11,12]),['winter','spring','summer','autumn']] = [0,0,0,1]

#change variable of season features from float to int
train = train.astype(
    {
        'winter': int, 'summer': int, 'spring': int, 'autumn': int
    }
)

#Create dummy variables for  Valencia_wind_deg & Seville_pressure
dummies_df = pd.get_dummies(train[['Valencia_wind_deg','Seville_pressure']], drop_first = True)

train = pd.concat([train, dummies_df], axis='columns')

#Drop original Valencia_wind_deg & Seville_pressure
train = train.drop(['Valencia_wind_deg', 'Seville_pressure' ], axis='columns')

# Re-organize the columns to have load_shortfall_3h at the end
column_titles = [col for col in train.columns if col!= 'load_shortfall_3h'] + ['load_shortfall_3h']
train = train.reindex(columns = column_titles)




y_train = train[['load_shortfall_3h']]
X_train = train[[col for col in train.columns if col not in 'load_shortfall_3h']]

# Fit model
lm_regression = LinearRegression(normalize=True)
print ("Training Model...")
lm_regression.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/rfr_model_25_300.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(lm_regression, open(save_path,'wb'))
