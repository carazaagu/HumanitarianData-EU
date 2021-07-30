# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


# Base
import pandas as pd
import numpy as np
import json

# Warnings
import warnings
warnings.filterwarnings("ignore")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import ipywidgets as widgets
from ipywidgets import interact
import folium as fl
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html

# Image
import base64

# Time
from datetime import date

# Clustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# Dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Sklearn preprocessing
from sklearn.preprocessing import OneHotEncoder

# Sklearn model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# Sklearn Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

# Sklearn Oversampling
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, InstanceHardnessThreshold
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

# Sklearn Parameter tunning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Sklearn Pipeline
from imblearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df2_gb = pd.read_csv('Data_dash/dataset_groupby.csv').drop('Unnamed: 0', axis=1)

dictionary = {'1. No poverty': ['Poverty ratio (% Pop.)',
'Vulnerable empolyment (% Empl.)',
'Salaried workers (% Empl.)'],

'2. Zero hunger': ['Undernourishment (% Pop.)',
'Crop production index',
'Obesity (% Pop. 18+)',
'Anemia among children (% Pop. 5-)'],

'3. Good health and wellbeing': ['Life expectancy (Years)',
'Birth rate (per 1,000)',
'Hospital beds (per 1,000)',
'Health expenditure (% GDP)',
'Government healt expenditure (% Health Exp.)',
'Suicide rate (per 100,000)',
'Population age 65+ (% Pop.)'],

'4. Quality education': ['Completed lower secondary (% Pop. 25+)',
'PISA: Mean performance average',
'Education expenditure (% Public Insitutions Exp.)',
'Education expenditure (% GDP)'],

'5. Gender equality': ['Female population (% Pop.)',
'School enrollment, tertiary (GPI)',
'Female labor force (% Labor)',
'Domestic violence legislation',
'No work discrimination legislation'],

'6. Clean water and sanitation' : ['Sanitation services (% Pop.)',
'Drinking water services (% Pop.)'],

'7. Affordable and clean energy' : ['Access to electricity (% Pop.)',
'Energy depletion (% GNI)'],

'8. Decent work and economic growth' : ['Unemployment rate (% Labor)',
'Youth not in education or employment (% Youth)',
'Part time employment (% Empl.)',
'Net national savings (% GDP)',
'Central government debt (% GDP)',
'GDP per capita (US$)',
'GNI per capita growth (%)',
'Account balance (% GDP)',
'Inflation (%)'],

'9. Industry, innovation and infrastructure' : ['Industry (% GDP)',
'R&D expenditure (% GDP)'],

'10. Reduced inequalities' : ['Refugee population per 1,000',
'Gini index'],

'11. Sustainable cities and communities' : ['Population living in slums (% Urban Pop.)',
'Urban population (% Pop.)',
'Exposure to pollution (% Pop.)'],

'12. Responsible consumption and production' : ['Particulate emission damage (% GNI)',
'Renewable energy consumption (% Total Consum)'],

'13. Climate action' : ['Greenhouse gas emissions (kt per 1,000)',
'CO2 emissions (kt per 1,000)',
'Carbon dioxide damage (% GNI)'],

'14. Life below' : ['Marine protected areas (% waters)',
'Fisheries production (metric tons per 1,000)',
'Aquaculture production (metric tons per 1,000)'],

'15. Life on land' : ['Agriculture, forestry, and fishing (% GDP)',
'Terrestrial protected areas (% of lands)'],

'16. Peace, justice and strong institutions' : ['Mortality in the road (per 100,000)',
'Children out of school (% primary school age)',
'Strength legal rights'],

'17. Partnership for the goals' : ['Net ODA provided (% GNI)',
'Fixed broadband subscriptions (per 100)',
'Trade (% GDP)',
'Foreign net direct investment (% GDP)']
}

# Dictionary to indicate how each indicator affects to the targets: positively or negatively

dictionary2={'Poverty ratio (% Pop.)': 'Negative',
'Vulnerable empolyment (% Empl.)': 'Negative',
'Salaried workers (% Empl.)': 'Positive',

'Undernourishment (% Pop.)': 'Negative',
'Crop production index': 'Positive',
'Obesity (% Pop. 18+)': 'Negative',
'Anemia among children (% Pop. 5-)': 'Negative',

'Life expectancy (Years)': 'Positive',
'Birth rate (per 1,000)': 'Positive',
'Hospital beds (per 1,000)': 'Positive',
'Health expenditure (% GDP)': 'Positive',
'Government healt expenditure (% Health Exp.)': 'Positive',
'Suicide rate (per 100,000)': 'Negative',
'Population age 65+ (% Pop.)': 'Negative',

'Completed lower secondary (% Pop. 25+)': 'Positive',
'PISA: Mean performance average': 'Positive',
'Education expenditure (% Public Insitutions Exp.)': 'Positive',
'Education expenditure (% GDP)': 'Positive',

'Female population (% Pop.)': 'Positive',
'School enrollment, tertiary (GPI)': 'Positive',
'Female labor force (% Labor)': 'Positive',
'Domestic violence legislation': 'Positive',
'No work discrimination legislation': 'Positive',

'Sanitation services (% Pop.)': 'Positive',
'Drinking water services (% Pop.)': 'Positive',

'Access to electricity (% Pop.)': 'Positive',
'Energy depletion (% GNI)': 'Negative',

'Unemployment rate (% Labor)': 'Negative',
'Youth not in education or employment (% Youth)': 'Negative',
'Part time employment (% Empl.)': 'Negative',
'Net national savings (% GDP)': 'Positive',
'Central government debt (% GDP)': 'Negative',
'GDP per capita (US$)': 'Positive',
'GNI per capita growth (%)': 'Positive',
'Account balance (% GDP)': 'Positive',
'Inflation (%)': 'Negative',

'Industry (% GDP)': 'Positive',
'R&D expenditure (% GDP)': 'Positive',

'Refugee population per 1,000': 'Positive',
'Gini index': 'Negative',

'Population living in slums (% Urban Pop.)': 'Negative',
'Urban population (% Pop.)': 'Negative',
'Exposure to pollution (% Pop.)': 'Negative',

'Particulate emission damage (% GNI)': 'Negative',
'Renewable energy consumption (% Total Consum)': 'Positive',

'Greenhouse gas emissions (kt per 1,000)': 'Negative',
'CO2 emissions (kt per 1,000)': 'Negative',
'Carbon dioxide damage (% GNI)': 'Negative',

'Marine protected areas (% waters)': 'Positive',
'Fisheries production (metric tons per 1,000)': 'Positive',
'Aquaculture production (metric tons per 1,000)': 'Positive',

'Agriculture, forestry, and fishing (% GDP)': 'Positive',
'Terrestrial protected areas (% of lands)': 'Positive',

'Mortality in the road (per 100,000)': 'Negative',
'Children out of school (% primary school age)': 'Negative',
'Strength legal rights': 'Positive',

'Net ODA provided (% GNI)': 'Positive',
'Fixed broadband subscriptions (per 100)': 'Positive',
'Trade (% GDP)': 'Positive',
'Foreign net direct investment (% GDP)':  'Positive'
}

# Encapsule variables and lists

indicators=list(df2_gb.columns[4:].sort_values())

countries=list(df2_gb['Country Name'].unique())

years=list(df2_gb['Year'].unique())

fig = px.scatter(df2_gb, x='Female labor force (% Labor)', y='Female population (% Pop.)', 
            animation_frame="Year",
            category_orders={'Country Name': countries, 'Year': years},
            hover_name="Country Name",
            color="Country Name",
            range_y=[df2_gb['Female population (% Pop.)'].min()-10,df2_gb['Female population (% Pop.)'].max()+10] if 'Female population (% Pop.)' != 'GDP per capita (current US$)' else [df2_gb['Female population (% Pop.)'].min() + 500,df2_gb['Female population (% Pop.)'].max()+ 1000],
            log_y=True if 'Female population (% Pop.)' == 'GDP per capita (current US$)' else False,
            size='Size',
            text='Country Name',
            opacity=0.33,
            hover_data={'Country Name':False, 'Year':True, 'Size':False, 'Female labor force (% Labor)':':.2f', 'Female population (% Pop.)':':.2f'},
            range_x=[df2_gb['Female labor force (% Labor)'].min()-10,df2_gb['Female labor force (% Labor)'].max()+10] if 'Female labor force (% Labor)' != 'GDP per capita (current US$)' else [df2_gb['Female labor force (% Labor)'].min() + 500,df2_gb['Female labor force (% Labor)'].max()+ 1000],
            template='ggplot2',
            height=745
            )

fig.update_layout(transition = {'duration': 20}, font_family='Trebuchet MS', font_size=9, hoverlabel_font_size=10)

fig2 = px.bar_polar(df2_gb, r='Greenhouse gas emissions (kt per 1,000)', theta="Country ISO3",
                       color='Greenhouse gas emissions (kt per 1,000)',
                       template="ggplot2",
                       color_continuous_scale= px.colors.sequential.Greens if dictionary2['Greenhouse gas emissions (kt per 1,000)'] == 'Positive' else px.colors.sequential.Reds,
                       range_color=[df2_gb['Greenhouse gas emissions (kt per 1,000)'].min(),df2_gb['Greenhouse gas emissions (kt per 1,000)'].max()],
                       range_r=[df2_gb['Greenhouse gas emissions (kt per 1,000)'].min(),df2_gb['Greenhouse gas emissions (kt per 1,000)'].max()],
                       hover_name='Country Name',
                       hover_data={'Year':True, 'Greenhouse gas emissions (kt per 1,000)':':.2f', 'Country ISO3':False},
                       animation_frame='Year',
                       height=600)

fig2.update_layout(font_family='Trebuchet MS', font_size=9, hoverlabel_font_size=10)    
fig2.update_coloraxes(colorbar_len=0.7)
fig2.update_coloraxes(colorbar_ticks="")
fig2.update_polars(radialaxis_visible=False)
fig2.update_coloraxes(colorbar_outlinewidth=0.21)
fig2.update_coloraxes(colorbar_thickness=30)
fig2.update_coloraxes(colorbar_title_text='Indicator Range')
fig2.update_coloraxes(colorbar_title_side='right')
fig2.update_coloraxes(colorbar_ticklabelposition='inside')


app.layout = html.Div(children=[
    html.H1(children=('Sustainable Development Goals - European Union'),
    style={'font-size':'44px'}
    ),

    html.Div(children=[
        html.P('''
        The world is evloving in a constant and exponential pace.'''),
    
        html.P('''It was many years ago when the world was transformed from a, hundred of independent units (empires, nations, countries, etc) to just one unit. 
    In the 18th century the globalization involved a huge increase in the relations of these different units that coexisted in this big world, up to a point that 
    there has been a global and worldwide integration.'''),

        html.P('''Therefore, countries and governments should consider the globalization not only as a matter of growth increase by more movement of people, 
    raw material or financial stocks, but also as a way to coordinate themselves in orther to face the current and future challenges that the humanity has and 
    will have to face. Among many challenges such as Artificial Intelligence or new scope for the labor market, the oldest and probably the least interesting 
    for governments has been the Equality and Social and Sustainable Development.'''),
    ]),
    html.Div(children=[

        html.P(children=[
        html.Img(src=app.get_asset_url('4.jpeg'),
        height='266px'
        )],
        style={'width':'21%', 'float':'right', 'textAlign':'right'}
        ),

        html.P(children=[
        html.Img(src=app.get_asset_url('4.jpeg'),
        height='266px'
        )],
        style={'width':'21%', 'float':'right', 'textAlign':'right'}
        ),

        html.P(children=[
        html.Img(src=app.get_asset_url('4.jpeg'),
        height='266px'
        )],
        style={'width':'21%', 'float':'right', 'textAlign':'right'}
        ),

        html.P(children=[
        html.Img(src=app.get_asset_url('2.png'),
        height='350px'
        )],
        style={'width':'76%', 'float':'right', 'textAlign':'right'}
        ),

        html.Br(),
        html.Br(),

        dcc.Markdown('''These ***17 Sustainable Development Goals*** have been set so that the world should achieve them as a coordinated organization:'''),
        html.Ol('1. No poverty'),
        html.Ol('2. Zero hunger'),
        html.Ol('3. Good health and wellbeing'),
        html.Ol('4. Quality education'),
        html.Ol('5. Gender equality'),
        html.Ol('6. Clean water and sanitation'),
        html.Ol('7. Affordable and clean energy'),
        html.Ol('8. Decent work and economic growth'),
        html.Ol('9. Industry, innovation and infrastructure'),
        html.Ol('10. Reduced inequalities'),
        html.Ol('11. Sustainable cities and communities'),
        html.Ol('12. Responsible consumption and production'),
        html.Ol('13. Climate action'),
        html.Ol('14. Life below'),
        html.Ol('15. Life on land'),
        html.Ol('16. Peace, justice and strong institutions'),
        html.Ol('17. Partnership for the goals')
        ,
    ]),
    html.Div('Hi', style={'height': '100px','width':'100%'}),


    
    html.Div(children=[
        
        html.Label(['Select Indicator X:', dcc.Dropdown(placeholder='Indicator X:',
            options=[{'label': 'Greenhouse gas emissions (kt per 1,000)', 'value' : 'Greenhouse gas emissions (kt per 1,000)'},
                {'label':'CO2 emissions (kt per 1,000)', 'value': 'CO2 emissions (kt per 1,000)'},
                {'label':'Carbon dioxide damage (% GNI)', 'value': 'Carbon dioxide damage (% GNI)'}],
            value='CO2 emissions (kt per 1,000)')],
            style={'width': '29%', 'height':'60px', 'float':'left','font-size':'11px','font-weight':'bold'}
        ),

        html.Label(
            style={'width': '2%', 'height':'60px', 'float':'left'}
        ),
        
        html.Label(['Select Indicator Y:', dcc.Dropdown(placeholder='Indicator Y:',
            options=[{'label': 'Greenhouse gas emissions (kt per 1,000)', 'value' : 'Greenhouse gas emissions (kt per 1,000)'},
                {'label':'CO2 emissions (kt per 1,000)', 'value': 'CO2 emissions (kt per 1,000)'},
                {'label':'Carbon dioxide damage (% GNI)', 'value': 'Carbon dioxide damage (% GNI)'}],
            value='Greenhouse gas emissions (kt per 1,000)')],
            style={'width': '29%', 'height':'60px', 'float':'left','font-size':'11px', 'font-weight':'bold'}
        ),

        dcc.Graph(
            id='scatterplot',
            figure=fig,
            style={'width': '60%', 'float':'left'}
        ),

         
        html.P('''Whatever I want to write.''',
            style={'height': '805px'} 
        ),

    ], style={'width':'100%'}),
    
    html.Div('Hello World', style={'height': '50px','width':'100%'}),

       
    html.Div(children=[

        html.Label(
            style={'width': '15%', 'height':'60px', 'float':'right'}
        ),

        html.Label(['Select Indicator:', dcc.Dropdown(placeholder='Indicator X:',
            options=[{'label': 'Greenhouse gas emissions (kt per 1,000)', 'value' : 'Greenhouse gas emissions (kt per 1,000)'},
                {'label':'CO2 emissions (kt per 1,000)', 'value': 'CO2 emissions (kt per 1,000)'},
                {'label':'Carbon dioxide damage (% GNI)', 'value': 'Carbon dioxide damage (% GNI)'}],
            value='CO2 emissions (kt per 1,000)')],
            style={'width': '30%', 'height':'60px', 'float':'right', 'font-size':'11px', 'font-weight':'bold'}
        ),

        html.Label(
            style={'width': '15%', 'height':'60px', 'float':'right'}
        ),

        
        dcc.Graph(
            id='barpolarplot',
            figure=fig2,
            style={'width': '60%', 'float':'right'}
        ),

        html.P('''Whatever I want to write.''',
            style={'height':'777px'}),

    ],style={'width':'100%'}, 
    )
], style={'font-family':'Trebuchet MS', 'padding-left':'88px', 'padding-right':'88px', 'padding-top':'33px'})

if __name__ == '__main__':
    app.run_server(debug=True)

