import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df=pd.read_csv('uber-eats-deliveries.csv')

df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Replace 'NaN ','conditions NaN' with np.nan
df.replace(['NaN ', 'NaN', 'conditions NaN'], np.nan, inplace=True)

for i in ['Delivery_person_Age','Delivery_person_Ratings','multiple_deliveries']:
  df[i] = df[i].astype(float)

def  get_numerical_summary(df):
  total_rows=df.shape[0]
  missing_cols=[col for col in df.columns if df[col].isnull().sum()>0]
  missing_percent={}
  for col in missing_cols:
    null_count=df[col].isnull().sum()
    per=(null_count/total_rows)*100
    missing_percent[col]=per
    print(f'{col}:{null_count},({round(per,3)})')
  return missing_percent

for idx in range(df.shape[0]):
  df.loc[idx,'missing_count']=df.iloc[idx,:].isnull().sum()

threshold=5
df.drop(df[df['missing_count']>threshold].index,axis=0,inplace=True)

for i in ['Delivery_person_Age','Delivery_person_Ratings','multiple_deliveries']:
  df[i].fillna(df[i].median(), inplace=True)

df = df.dropna(subset=['Time_Orderd'])

for i in ['Weatherconditions', 'Road_traffic_density','Festival', 'City']:
  mod=df[i].mode()[0]
  df[i].fillna(mod, inplace=True)

df['order_datetime'] = df['Order_Date'] + ' ' + df['Time_Orderd']
df['pick_datetime'] = df['Order_Date'] + ' ' + df['Time_Order_picked']


df['order_datetime'] = pd.to_datetime(df['order_datetime'],format='%d-%m-%Y %H:%M:%S')
df['pick_datetime'] = pd.to_datetime(df['pick_datetime'],format='%d-%m-%Y %H:%M:%S')

df.drop(columns=['Order_Date', 'Time_Orderd', 'Time_Order_picked','missing_count'], inplace=True)

from datetime import timedelta
def calculate_time_difference(order_time, pick_time):
    if pick_time < order_time:
        pick_time += timedelta(days=1)
    return (pick_time - order_time).total_seconds() / 60.0

df['Order_Preparation_Time'] = df.apply(lambda row: calculate_time_difference(row['order_datetime'], row['pick_datetime']), axis=1)

# create another column
def categorize_time_of_day(time):
    if time.hour >= 5 and time.hour < 12:
        return 'Morning'
    elif time.hour >= 12 and time.hour < 17:
        return 'Afternoon'
    elif time.hour >= 17 and time.hour < 21:
        return 'Evening'
    else:
        return 'Night'

# Apply the function to create a new column
df['Time_of_Day'] = df['order_datetime'].apply(categorize_time_of_day)

df['Time_taken(min)'] = df['Time_taken(min)'].str.strip('(min) ').astype(int)

df['Weatherconditions'] = df['Weatherconditions'].str.strip('conditions ')

df.drop(['ID','Delivery_person_ID'],axis=1,inplace=True)

df['Restaurant_latitude'] = df['Restaurant_latitude'].abs()

def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    r = 6371  # Radius of Earth in kilometers
    distance = r * c

    return distance

df['distance_km'] = df.apply(lambda row: haversine(row['Restaurant_latitude'], row['Restaurant_longitude'],
                                                   row['Delivery_location_latitude'], row['Delivery_location_longitude']),
                            axis=1)

df['distance_km'] = df['distance_km'].round(3)

df.drop(['Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude'],axis=1,inplace = True)

df['order_day'] = df['order_datetime'].dt.day
df['order_hour'] = df['order_datetime'].dt.hour
df['order_minute'] = df['order_datetime'].dt.minute

df.drop(['order_datetime','pick_datetime'],axis=1,inplace=True)

Q1_ratings = df['Delivery_person_Ratings'].quantile(0.25)
Q3_ratings = df['Delivery_person_Ratings'].quantile(0.75)
IQR_ratings = Q3_ratings - Q1_ratings

lower_bound_ratings = Q1_ratings - 1.5 * IQR_ratings
upper_bound_ratings = Q3_ratings + 1.5 * IQR_ratings

df_encoded = pd.get_dummies(df, columns=['Weatherconditions','Road_traffic_density','Type_of_order','Type_of_vehicle','Festival', 'City','Time_of_Day']).astype(int)

df_final = df_encoded.copy()

x=df_final.drop('Time_taken(min)',axis=1)
y=df_final['Time_taken(min)']

from imblearn.over_sampling import SMOTE

# Let's assume the column 'Road_traffic_density_High' is what we're focusing on
# Creating a SMOTE object
smote = SMOTE(sampling_strategy='not minority', random_state=42)

# Perform SMOTE on the entire dataset
x_resampled, y_resampled = smote.fit_resample(x, y)

# Convert back to DataFrame
x_resampled = pd.DataFrame(x_resampled, columns=x.columns)
y_resampled = pd.Series(y_resampled, name='Time_taken(min)')

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

#import lightgbm as lgb
#model = lgb.LGBMRegressor(force_row_wise=True)
#model.fit(x_train, y_train)
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

base_estimator = DecisionTreeRegressor()  # You can adjust max_depth as needed
ada_regressor = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=50, random_state=42)

# Train the model
ada_regressor.fit(x_train, y_train)

with open('model.pkl','wb') as model_files:
  pickle.dump(ada_regressor,model_files)