import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import sys

global decoded_dict
decoded_dict = {}
decoded_dict['Corrected_obesity_level_type'] = {0: 'Insufficient Weight', 1: 'Normal Weight', 2: 'Overweight', 3: 'Obesity Type I', 4: 'Obesity Type II', 5: 'Obesity Type III'}

def GetCategory(mbi):
  if mbi < 18.5: return 0
  elif mbi < 25: return 1
  elif mbi < 30: return 2
  elif mbi < 35: return 3
  elif mbi < 40: return 4
  else: return 5

def transforme_dataframe(df):
    df['Age'] = df['Age'].astype('uint8')
    df['Height'] = df['Height'].round(2)
    df['MBI'] = df['Weight'] / df['Height']**2
    df['Corrected_obesity_level_category'] = df['MBI'].apply(GetCategory).astype('uint8')
    #Using LabelEncoder for the rest of the object type columns
    encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype=='object':
            encoded_labels = encoder.fit_transform(df[col])
            df[col] = encoded_labels
            decoded_dict[col] = {int(label): value for label, value in zip(encoded_labels, encoder.inverse_transform(encoded_labels))}
            df[col] = df[col].astype('uint8')
    decoded_dict['Age'] = 'Numeric values'
    decoded_dict['Height'] = 'Numeric values'
    decoded_dict['Frequency_eat_vegetables'] = {1: 'Never', 2: 'Sometimes', 3: 'Always'}
    decoded_dict['Frequency_water'] = {1: 'Less than 1L/day', 2: 'Between 1 and 2L/day', 3: 'More than 2L/day'}
    decoded_dict['Frequency_physical_activity'] = {0: 'No activity', 1: '1 or 2 days/week', 2: '2 or 4 days/week', 3: '4+ days/week'}
    decoded_dict['Time_using_technology_devices'] = {0: '0-2 hours/day', 1: '3-5 hours/day', 2: '5+ hours/day'}
    return df 

def train_model():
    df = pd.read_csv("ObesityDataSet.csv", sep=',')
    df.drop(columns=['MTRANS', 'SMOKE', 'NCP', 'NObeyesdad'], inplace=True)
    df.columns = ['Gender', 'Age', 'Height', 'Weight', 'Family_history_with_overweight', 'Frequency_eat_high_caloric_food', 'Frequency_eat_vegetables',
              'Frequency_eat_between_meals', 'Frequency_water', 'Monitoring_calories_consumption', 'Frequency_physical_activity', 'Time_using_technology_devices',
              'Frequency_alcohol']
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    #prepare the data
    #train model
    df = transforme_dataframe(df)
    
    #normalize
    X = df.drop(columns = ['MBI', 'Corrected_obesity_level_category', 'Weight'])
    scaler = MinMaxScaler()
    scaler.fit(X)
    x_train_scaled = scaler.transform(X)
    
    y = df['Corrected_obesity_level_category']

    rf_model = RandomForestClassifier(
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=50
    )
    rf_model.fit(x_train_scaled, y)
    return rf_model, scaler

def predict_obesity(person_dict, model, scaler):
    person_dict_df = pd.DataFrame(person_dict)
    person_dict_df_scaled = scaler.transform(person_dict_df)
    person_dict_prediction = model.predict(person_dict_df_scaled)[0]
    predicted_result = decoded_dict['Corrected_obesity_level_type'][person_dict_prediction]
    return predicted_result

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
from datetime import datetime

model, scaler = train_model()
model_seed = random.random() * 100

app = FastAPI()

@app.get('/')
async def root():
    return {"message": "Hello world"}

@app.get('/predict')
async def predict(gender=None, age=None, height=None, family_history_with_overweight=None,
                    frequency_eat_high_caloric_food=None, frequency_eat_vegetables=None,
                    frequency_eat_between_meals=None, frequency_water=None,
                    monitoring_calories_consumption=None, frequency_physical_activity=None,
                    time_using_technology_devices=None, frequency_alcohol=None, 
                    name=None):
     
    if gender==None or age==None or height==None or family_history_with_overweight==None or frequency_eat_high_caloric_food==None or frequency_eat_vegetables==None or frequency_eat_between_meals==None or frequency_water==None or monitoring_calories_consumption==None or frequency_physical_activity==None or time_using_technology_devices==None or frequency_alcohol==None:
        message = "Error: There are some missing values !"
        content = f"<h1>{message}</h1>\n"
        for col in decoded_dict:
            if col!="Corrected_obesity_level_category": content += f"<pre>{col} : {decoded_dict[col]}</pre>\n"
        return HTMLResponse(content=content)
        
    else:
        person_dict = {'Gender': [gender], 'Age': [age], 'Height': [height], 'Family_history_with_overweight': [family_history_with_overweight],
                 'Frequency_eat_high_caloric_food': [frequency_eat_high_caloric_food], 'Frequency_eat_vegetables': [frequency_eat_vegetables], 'Frequency_eat_between_meals': [frequency_eat_between_meals],
                 'Frequency_water': [frequency_water], 'Monitoring_calories_consumption': [monitoring_calories_consumption], 'Frequency_physical_activity': [frequency_physical_activity],
                 'Time_using_technology_devices': [time_using_technology_devices], 'Frequency_alcohol': [frequency_alcohol]}
        
        predicted_obesity = predict_obesity(person_dict, model, scaler)
        
        new_dict = {'Gender':decoded_dict['Gender'][int(gender)],
                    'Age':age,
                    'Height':height,
                    'Family history with overweight':decoded_dict['Family_history_with_overweight'][int(family_history_with_overweight)],
                    'Eating high calory food':decoded_dict['Frequency_eat_high_caloric_food'][int(frequency_eat_high_caloric_food)],
                    'Frequency of eating vegetables':decoded_dict["Frequency_eat_vegetables"][int(frequency_eat_vegetables)],
                    'Frequency of eating between meals':decoded_dict['Frequency_eat_between_meals'][int(frequency_eat_between_meals)],
                    'Amount of water drank / day':decoded_dict['Frequency_water'][int(frequency_water)],
                    'Monitoring calories consumption':decoded_dict['Monitoring_calories_consumption'][int(monitoring_calories_consumption)],
                    'Amount of physical activity / week':decoded_dict['Frequency_physical_activity'][int(frequency_physical_activity)],
                    'Time using a technological devide / day':decoded_dict['Time_using_technology_devices'][int(time_using_technology_devices)],
                    'Frequency of drinking alcohol':decoded_dict['Frequency_alcohol'][int(frequency_alcohol)],
                    'Predicted obesity type': predicted_obesity,
                    'Accuracy': 0.83,
                    'Date of prediction': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Model seed': model_seed
                    }
        
        if name!=None: new_dict['Name'] = name
    
        json_data = jsonable_encoder(new_dict)
    
        return json_data