import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import dash
from dash import Dash, dcc, html, Input, Output, State, callback
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
MODEL = None
SCALER = None

def transforme_dataframe(df):
    df['Age'] = df['Age'].astype('uint8')
    df['Height'] = df['Height'].round(2)

    #Using LabelEncoder for the rest of the object type columns
    encoder = LabelEncoder()
    decoded_dict = {}
    for col in df.columns:
        if df[col].dtype=='object':
            encoded_labels = encoder.fit_transform(df[col])
            df[col] = encoded_labels
            decoded_dict[col] = {label: value for label, value in zip(encoded_labels, encoder.inverse_transform(encoded_labels))}
            df[col] = df[col].astype('uint8')
       
    return df

def train_model():
    df = pd.read_csv("ObesityDataSet.csv", sep=',')
    df.drop(columns=['Weight', 'MTRANS', 'SMOKE', 'NCP'], inplace=True)
    df.columns = ['Gender', 'Age', 'Height', 'Family_history_with_overweight', 'Frequency_eat_high_caloric_food', 'Frequency_eat_vegetables',
              'Frequency_eat_between_meals', 'Frequency_water', 'Monitoring_calories_consumption', 'Frequency_physical_activity', 'Time_using_technology_devices',
              'Frequency_alcohol', 'Obesity_level_category']
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    #prepare the data
    #train model
    df = transforme_dataframe(df)
    
    #normalize
    X = df.drop(columns = ['Obesity_level_category'])
    scaler = MinMaxScaler()
    scaler.fit(X)
    x_train_scaled = scaler.transform(X)
    
    y = df['Obesity_level_category']


    rf_model = RandomForestClassifier(
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=50
    )
    rf_model.fit(x_train_scaled, y)
    return rf_model, scaler

# Create Dash app
app = dash.Dash(__name__)

# Define layout
try:
    app.layout = html.Div([
        html.Header([
            html.H1("Python for data analysis : Final project"),
            html.H2("Subject : Estimation of obesity levels based on eating habits and physical condition"),
            html.P("Team members : Louis MARTYR, Killian LAFAYE, Marc LEMAISTRE", id="Team-members-:-Louis-MARTYR,-Killian-LAFAYE,-Marc-LEMAISTRE"),
        ]),
        html.Div([
            html.Img(
                src="./assets/cannard.jpeg"
            ),
            html.Div([
                html.Label('Gender: '),
                dcc.Dropdown(
                    ['Female', 'Male'],
                    'Female',
                    clearable=False,
                    id='Gender'
                ),
            ]),
            html.Div([
                html.Label('Age (y): '),
                dcc.Input(id='Age', type='number'),
            ]),
            html.Div([
                html.Label('Height (m): '),
                dcc.Input(id='Height', type='number'),
            ]),
            html.Div([
                html.Label('Family history with overweight: '),
                dcc.Dropdown(
                    ['yes', 'no'],
                    'no',
                    clearable=False,
                    id='Family_history_with_overweight'
                ),
            ]),
            html.Div([
                html.Label('Do you frequently eat high caloric food: '),
                dcc.Dropdown(
                    ['yes', 'no'],
                    'no',
                    clearable=False,
                    id='Frequency_eat_high_caloric_food'
                ),
            ]),
            html.Div([
                html.Label('Frequency you eat vegetables: '),
                dcc.Dropdown(
                    ['Never', 'Sometimes', 'Always'],
                    'Never',
                    id='Frequency_eat_vegetables',
                ),
            ]),
            html.Div([
                html.Label('Frequency you eat between meals: '),
                dcc.Dropdown(
                    ['Never','Sometimes','Frequently','Always'],
                    'no',
                    id='Frequency_eat_between_meals'
                ),
            ]),
            html.Div([
                html.Label('How much water do you drink a day: '),
                dcc.Dropdown(
                    ['Less than 1L/day', 'Between 1 and 2L/day', 'More than 2L/day'],
                    'Between 1 and 2L/day',
                    id='Frequency_water'
                ),
            ]),
            html.Div([
                html.Label('Do you monitore you calorie consumption: '),
                dcc.Dropdown(
                    ['yes', 'no'],
                    'no',
                    clearable=False,
                    id='Monitoring_calories_consumption'
                ),
            ]),
            html.Div([
                html.Label('How much day do you execise a week: '),
                dcc.Dropdown(
                    ['Never', '1 or 2', '2 or 4', '4 or more'],
                    'Never',
                    id='Frequency_physical_activity',
                ),
            ]),
            html.Div([
                html.Label('Time you use technology devices per day: '),
                dcc.Dropdown(
                    ['less than 2 hours', 'between 3-5 hours', 'more than 5 hours'],
                    'between 3-5 hours',
                    id='Time_using_technology_devices',
                ),
            ]),
            html.Div([
                html.Label('Frequency you drink alcohol: '),
                dcc.Dropdown(
                    ['Never', 'Sometimes', 'Frequently', 'Always'],
                    'Sometimes',
                    id='Frequency_alcohol',
                ),
            ]),
            html.Button('Submit', id='submit-val', n_clicks=0),
            html.Div(id="output-div"),
        ]),
        html.Div([
            html.H2("I. Introduction"),
            html.Iframe(srcDoc=open('./HTML_contents/introduction.html', 'r').read(), width='100%', height='500px'),
        ]),
        html.Div([
            html.H2("II. Data processing"),
            html.Iframe(srcDoc=open('./HTML_contents/data_processing.html', 'r').read(), width='100%', height='500px'),
        ]),
        html.Div([
            html.H2("III. Data Visualization & Analysis"),
            html.Iframe(srcDoc=open('./HTML_contents/data_vis.html', 'r').read(), width='100%', height='500px'),
        ]),
        html.Div([
            html.H2("IV. Data distribution"),
            html.Iframe(srcDoc=open('./HTML_contents/data_distribution.html', 'r').read(), width='100%', height='500px'),
        ]),
        html.Div([
            html.H2("V. Correlation handling"),
            html.Iframe(srcDoc=open('./HTML_contents/correlation.html', 'r').read(), width='100%', height='500px'),
        ]),
        html.Div([
            html.H2("VI. Prediction model"),
            html.Iframe(srcDoc=open('./HTML_contents/model.html', 'r').read(), width='100%', height='500px'),
        ]),
        html.Div([
            html.H2("VII. Conclusion"),
            html.Iframe(srcDoc=open('./HTML_contents/conclusion.html', 'r').read(), width='100%', height='500px'),
        ]),
    
        # Using html.Iframe to load an HTML file
    
    
        # Other components as needed
    ])
except FileNotFoundError as e:
    print(e.strerror)
    print("Please make sure that your working directory is the same as the directory of the running file, or that the 'HTML_contents' directory is located in the same directory as this program.")

@callback(
    Output('output-div', 'children'),
    Input('submit-val', 'n_clicks'),
    State('Gender', 'value'),
    State('Age', 'value'),
    State('Height', 'value'),
    State('Family_history_with_overweight', 'value'),
    State('Frequency_eat_high_caloric_food', 'value'),
    State('Frequency_eat_vegetables', 'value'),
    State('Frequency_eat_between_meals', 'value'),
    State('Frequency_water', 'value'),
    State('Monitoring_calories_consumption', 'value'),
    State('Frequency_physical_activity', 'value'),
    State('Time_using_technology_devices', 'value'),
    State('Frequency_alcohol', 'value'),

)
def update_output(n_clicks, Gender, Age, Height, Family_history_with_overweight, Frequency_eat_high_caloric_food, Frequency_eat_vegetables,
                Frequency_eat_between_meals, Frequency_water, Monitoring_calories_consumption, Frequency_physical_activity, Time_using_technology_devices,
                Frequency_alcohol):
    if n_clicks<=0:
        return ''

    #predicting the user
    dict_user={
        'Gender': [Gender],
        'Age': [Age],
        'Height': [Height],
        'Family_history_with_overweight': [Family_history_with_overweight],
        'Frequency_eat_high_caloric_food': [Frequency_eat_high_caloric_food],
        'Frequency_eat_vegetables': [Frequency_eat_vegetables],
        'Frequency_eat_between_meals': [Frequency_eat_between_meals],
        'Frequency_water': [Frequency_water],
        'Monitoring_calories_consumption': [Monitoring_calories_consumption],
        'Frequency_physical_activity': [Frequency_physical_activity],
        'Time_using_technology_devices': [Time_using_technology_devices],
        'Frequency_alcohol': [Frequency_alcohol],
    }
    df_user = pd.DataFrame(dict_user)
        
    df_user = transforme_dataframe(df_user)

    x_user = SCALER.transform(df_user)
    y_user = MODEL.predict(x_user)


    dict_obesity = {0: 'Insufficient Weight', 1: 'Normal Weight', 2: 'Overweight', 3: 'Obesity Type I', 4: 'Obesity Type II', 5: 'Obesity Type III'}
    id = random.random()
    return f'Vous êtes de catégorie: {dict_obesity[y_user[0]]}, \nEssai numéro: {id}'

if __name__ == '__main__':
    MODEL, SCALER = train_model()
    app.run_server(debug=True)