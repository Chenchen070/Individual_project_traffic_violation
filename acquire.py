import pandas as pd
import numpy as np

def acquire_data():
    df = pd.read_csv('Traffic_Violations_montgomery_county.csv')
    return df