import pandas as pd
import datetime as dt
import numpy as np
from sklearn.model_selection import train_test_split
import math

def clean_data(df):
    # drop columns
    drop_col = ['SeqID', 'Agency', 'Accident', 'Fatal', 'Commercial License','HAZMAT', 'Commercial Vehicle', 
                'Work Zone', 'State', 'VehicleType', 'Year', 'Make', 'Model', 'Color', 'Charge', 'Article', 
                'Driver State', 'DL State', 'Search Reason For Stop', 'Search Arrest Reason', 'Location',
                'Search Conducted', 'Search Disposition', 'Search Outcome', 'Search Reason', 'Search Type',
                'Geolocation','Latitude', 'Longitude','Description', 'Driver City', 'Arrest Type']
    df = df.drop(columns = drop_col)
    
    # # convert time format
    df['date_time'] = df['Date Of Stop'] + ' ' + df['Time Of Stop']
    col = ['Date Of Stop', 'Time Of Stop']
    df = df.drop(columns = col)
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # set the time to index and create new columns for year, month, day of the week and hour
    df.set_index('date_time', inplace=True)
    df.sort_index(inplace=True)
    df['month'] = df.index.strftime('%m-%b')
    df['day_of_week'] = df.index.strftime('%A')
    df['year'] = df.index.strftime('%Y')
    df['hour'] = df.index.strftime('%H')
    
    # convert the boolean value into int
    df['Contributed To Accident'] = np.where(df['Contributed To Accident'] == True, 1, df['Contributed To Accident'])
    df['Contributed To Accident'] = np.where(df['Contributed To Accident'] == False, 0, df['Contributed To Accident'])

    # SubAgency S15 and W15 only have less than 10 rows.
    df = df[df.SubAgency != 'S15']
    df = df[df.SubAgency != 'W15']
    
    # get all the data related to accident
    df = df[df['Contributed To Accident'] == 1]
    
    return df

def split_data(df):
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)

    train, validate = train_test_split(train_validate, test_size=.3, 
                                       random_state=123)
    
    return train, validate, test

def prep_data(df):
    
    df = clean_data(df)
    train, validate, test = split_data(df)
    return train, validate, test

from sklearn.metrics import confusion_matrix
def f1_score(y, prediction):
    TN, FP, FN, TP = confusion_matrix(y, prediction).ravel()
    ALL = TP + FP + FN + TN

    true_positive_rate = sensitivity = recal = power = TP/(TP+FN)
    precision = PPV = TP/(TP+FP)
    f1 = 2*(precision*recal)/(precision+recal)
    return f1