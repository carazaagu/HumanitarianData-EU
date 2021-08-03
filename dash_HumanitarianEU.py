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
from dash.dependencies import Input, Output

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
df2 = pd.read_csv('Data_dash/dataset_no_groupby.csv').drop('Unnamed: 0', axis=1)
df2_gb = pd.read_csv('Data_dash/dataset_groupby.csv').drop('Unnamed: 0', axis=1)
df2_gb_2 = pd.read_csv('Data_dash/dataset_groupby_2.csv').drop('Unnamed: 0', axis=1)


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
            height=500
            )

fig.update_layout(transition = {'duration': 20},  margin=dict(t=44),font_family='Arial', font_size=9, hoverlabel_font_size=10)

fig2 = px.bar_polar(df2_gb, r='Greenhouse gas emissions (kt per 1,000)', theta="Country ISO3",
                       color='Greenhouse gas emissions (kt per 1,000)',
                       template="ggplot2",
                       color_continuous_scale= px.colors.sequential.Greens if dictionary2['Greenhouse gas emissions (kt per 1,000)'] == 'Positive' else px.colors.sequential.Reds,
                       range_color=[df2_gb['Greenhouse gas emissions (kt per 1,000)'].min(),df2_gb['Greenhouse gas emissions (kt per 1,000)'].max()],
                       range_r=[df2_gb['Greenhouse gas emissions (kt per 1,000)'].min(),df2_gb['Greenhouse gas emissions (kt per 1,000)'].max()],
                       hover_name='Country Name',
                       hover_data={'Year':True, 'Greenhouse gas emissions (kt per 1,000)':':.2f', 'Country ISO3':False},
                       animation_frame='Year', height=500
                       )

fig2.update_layout(font_family='Arial', font_size=9, hoverlabel_font_size=10,transition = {'duration': 20})    
fig2.update_coloraxes(colorbar_len=0.7)
fig2.update_coloraxes(colorbar_ticks="")
fig2.update_polars(radialaxis_visible=False)
fig2.update_coloraxes(colorbar_outlinewidth=0.21)
fig2.update_coloraxes(colorbar_thickness=30)
fig2.update_coloraxes(colorbar_title_text='Indicator Range')
fig2.update_coloraxes(colorbar_title_side='right')
fig2.update_coloraxes(colorbar_ticklabelposition='inside')

new_df = df2[(df2['Year'] == 2019) & (df2['Vulnerable empolyment (% Empl.)'] !=0)]
new_df['Rank'] = new_df['Vulnerable empolyment (% Empl.)'].rank(ascending=False)
new_df['Average_EU'] = round(new_df['Vulnerable empolyment (% Empl.)'].mean(),2)

geo_objects = json.load(open('Data/europe.json'))


fig3=px.choropleth(new_df, geojson=geo_objects, featureidkey="properties.ISO3",locations='Country ISO3',
                      color_continuous_scale='Reds',
                      color='Vulnerable empolyment (% Empl.)', scope='europe',
                     projection='natural earth', template='ggplot2', height=500,
                     hover_name='Country Name',
                     hover_data={'Country ISO3':False, 'Vulnerable empolyment (% Empl.)':True, 'Average_EU':True, 'Rank':True})

fig3.update_layout(font_family='Arial', font_size=10, hoverlabel_font_size=10, margin=dict(l=0, r=0, t=0, b=0), hoverlabel_bgcolor='Green')
fig3.update_coloraxes(colorbar_len=0.7)
fig3.update_coloraxes(colorbar_ticks="")
fig3.update_coloraxes(colorbar_outlinewidth=0.21)
fig3.update_coloraxes(colorbar_thickness=30)
fig3.update_coloraxes(colorbar_title_text='Indicator Range')
fig3.update_coloraxes(colorbar_title_side='right')
fig3.update_coloraxes(colorbar_ticklabelposition='inside')

fig4 = px.histogram(df2_gb_2,
                   y="Country Name",
                   x='MinMax',
                   color='Country Name',
                   color_discrete_map={'Austria':'green', 'Belgium':'green','Bulgaria':'red', 'Croatia':'red', 'Cyprus':'red',
                   'Czech Republic':'red', 'Denmark':'green', 'Estonia':'red', 'Finland':'green', 'France':'green',
                   'Germany':'green', 'Greece':'red', 'Hungary':'red', 'Ireland':'green', 'Italy':'green', 'Latvia':'red',
                   'Lithuania':'red', 'Luxembourg':'green', 'Malta':'red', 'Netherlands':'green', 'Poland':'red', 'Portugal':'red',
                   'Romania':'red', 'Slovak Republic':'red', 'Slovenia':'red', 'Spain':'red', 'Sweden':'green'},
                   opacity=0.6,
                   hover_name='Country Name',
                   hover_data=['Rank'],
                   labels={'MinMax':'Index'},
                   orientation='h',
                   height=500,
                   range_x = [-1.1, 1.1],
                   template='ggplot2',
                   category_orders={'Country Name':['Luxembourg', 'Ireland', 'Denmark', 'Sweden', 'Netherlands', 'Austria',
                     'Finland', 'Belgium', 'Germany', 'France', 'Italy', 'Spain', 'Cyprus', 'Greece', 'Slovenia', 'Malta',
                     'Portugal', 'Czech Republic', 'Estonia', 'Slovak Republic', 'Hungary', 'Croatia', 'Lithuania',
                     'Latvia', 'Poland', 'Romania', 'Bulgaria']})

fig4.update_layout(showlegend=False,
                   xaxis_title="Dimensionality Reduction Index",
                   yaxis_title=None,
                   font_family='Arial',
                   font_size=10,
                   margin=dict(t=44, b=10, l=125))

im_indicator = app.get_asset_url('1. No poverty.jpg')

app = dash.Dash(__name__)

app.layout = html.Div(children=[

    html.P(children=[
        html.H3('''HUMANITARIAN DATA - EUROPEAN UNION
            '''),
        html.P('''Carlos Azagra - Ironhack Barcelona''')
        ],
        style={'height': '100px', 'width':'33%', 'marginRight':'1%', 'marginLeft':'1%', 'float':'right', 'textAlign':'right'} 
        ),
    
    html.P(children=[
    html.Img(src=app.get_asset_url('SDG_.png'),
        height='111px'
        ),
        ],
        style={'width':'67%'} 
        ),

    html.Div(children=[

        html.P(children=[

            html.P('''Progress towards reaching these goals was very different accross countries. And more importantly, results were far from being enough.
            ''', style={'marginTop':'0'}),

            html.P('''In 2015, the 8 Millenium Development Goals were overwritten by the 17 Sustainable Developments Goals within the United Nations Agenda 2030.
            These 17 Goals have a clear and specific vision of where the World should lead to and how to get there.'''),

            html.P(
            '''Despite the achievement of these Goals is not compulsory nor binding, UN considered that having a team monitoring the results 
            would be helpful to coordinate the actions, to point out which countries are acting more efficiently and therefore use them as a benchmark for 
            other countries that do not want or do not know how to approach the targets. Carrying out this duty is on the hands of the UN Statistical Comission,
            created to coordinate, track and monitor of the results obtained. '''),

            html.P('''
            This entity works with various repository agencies and stakeholders. In spite of being very complete, explicit and detailed reports, these files turn out 
            to be very long, technical, static and not interactive. In other words, hard to read and difficult to obtain conclusions, or to compare between countries. '''),

            html.P('''
            The main goal of this project is to provide the user a set of interactive tools than can help to monitor how European Union country members are coping with
            the SDG, by breaking each Goal down into relevant indicators, which can be analysed by year or country. These tools will allow the user to easily and visually
            draw conclusions, compare data between countries, analyse the evolution of the indicators and even compare relationships between them.''')
        ], style={'width':'48.5%', 'marginLeft':'1.5%', 'marginTop':'0','textAlign': 'justify', 'float':'right','height': '510px'}
        ),

        html.P(children=[
            html.P(
            '''The world is evolving at a constant and exponential pace.'''),
        
            html.P(
            '''It was many years ago when the world was transformed from hundreds of independent social structures (empires, nations, countries, etc) to just one unit. 
            In the 18th century the globalization involved a huge increase in the relations of these different units that coexisted in this big world, up to a point that 
            there has been a global and worldwide integration.'''),

            html.P(
            '''One of the lessons that COVID19 crisis has taught us is that the biggest and most threatening challenges need a global and
            coordinated response, as they cannot be approached individually.'''),

            html.P(
            '''Therefore, countries and governments should consider the globalization not only as a matter of growth increase (more movement of people, 
            raw materials or financial stocks) but also as a way to coordinate themselves in orther to face the current and future challenges that the humanity has and 
            will have to face. Among many challenges such as Artificial Intelligence or new scope for the labor market, making the world we are living in a better place for
            future generations should also be considered as a priority for each and every country. All governments should be focused on fostering policies oriented to reduce
            local and worldwide inequalities among populations, to develope an economic and social model that is sustainable, to preserve the enviornment, or to stand for 
            peace and justice.'''),

            html.P(
            '''As a first step to try to make a call for action and to coordinate the efforts from different countries, there was the necessity to
            define clear, measurable and time-specific goals to achieve as a way to define a clear path to pursue. That is why in 2000 all members from the United 
            Nations agreed to define 8 goals (Millenium Development Goals) to be achieved in 2015.'''),
        ], style={'width':'48.5%', 'marginRight':'1.5%', 'textAlign': 'justify','height': '510px'}
        )

    ], style={'height': '510px'}
    ),

    #html.Br(),

    html.Div(style={'height': '5px','width':'100%'}),

    html.Div(children=[

        html.P(children=[

            html.H2('''17 SUSTAINABLE DEVELOPMENT GOALS AND ITS INDICATORS''', style={'textAlign':'center'}),

            dcc.RadioItems(
                id='radio_items',
                options=[{'label':x, 'value' : x} for x in dictionary.keys()],
                value='1. No poverty',
                labelStyle={'display':'block', 'lineHeight':'1.44'})
                        
            ],style={'width':'33%', 'marginLeft':'1%', 'marginRight':'1%', 'float':'right'}),
        
        html.P(style={'height':'0.3%'}),

        html.P(children=[
            
            html.P(style={'height':'5%' if im_indicator == app.get_asset_url('8. Decent work and economic growth.jpg') else '1%'}),

            html.Img(src=app.get_asset_url('1. No poverty.jpg'),
            id='image_indicator',
            height='240px',
            style={'display':'block','margin':'auto'}
            ),

            html.Br(),

            dcc.Checklist(
                id='check_list_ind',
                options=[{'label':x, 'value' : x} for x in dictionary['1. No poverty']],
                value= [x for x in dictionary['1. No poverty']],
                labelStyle={'display':'block', 'lineHeight':'1.55', 'margin':'0 auto', 'textAlign':'center'}
                )

        ],
        style={'width':'64%', 'marginLeft':'1%', 'height':'500px', 'backgroundColor':'White'})

    ], style={'backgroundColor':'lightsteelblue', 'height':'530px'}),

    html.Br(),

    html.Div(style={'height': '10px','width':'100%'}),
    
    html.Div(children=[
        html.P(
            dcc.Graph(
            id='interactive_map',
            figure=fig3),
        style={'width':'64%', 'float':'right', 'marginRight':'1%'}),

        html.P(children=[
            
            html.Br(),
            html.Br(),

        html.H2('EU INTERACTIVE MAP', style={'textAlign':'center'}),
        

            html.Br(),
                        
            html.Label(['Select Goal:', dcc.Dropdown(id='Target',
                options=[{'label':x, 'value' : x} for x in dictionary.keys()],
                value='1. No poverty')]
                , style={'font-size':'11px', 'font-weight':'bold'},
                ),

            html.Label(['Select Indicator:', dcc.Dropdown(id='Indicator',
                options=[{'label':x, 'value' : x} for x in dictionary['1. No poverty']],
                value='Vulnerable empolyment (% Empl.)')]
                , style={'font-size':'11px', 'font-weight':'bold'},
                ),

            html.Label(['Select Year:', dcc.Dropdown(id='Year',
                options=[{'label':x, 'value' : x} for x in list(-np.sort(-df2[df2['Vulnerable empolyment (% Empl.)'] != 0]['Year'].unique()))],
                value=2019)]
                , style={'font-size':'11px', 'font-weight':'bold'},
                ),
            
            html.Br(),

            html.P('''By playing with the Map, you can select the Target, Indicator and Year you are interested in, to monitor the stats and quickly compare
            them among countries. '''),

            html.P('''You can interact with the Map by hovering the cursor over the countries: you will se the Country Name, the rate of the
            Indicator selected, the average of the Indicator in the EU, and the Ranking in which the country is spotted''')
            
            ],style={'width': '33%', 'height':'500px', 'marginLeft':'1%', 'marginRight':'1%'}),
            
    ],style={'width':'100%','height':'530px','backgroundColor':'lightsteelblue'}),
    
    html.Br(),

    html.Div(style={'height': '10px','width':'100%'}),

    html.Div(children=[

        html.P(style={'height':'0.3%'}),
        
        html.P(children=[

            html.Br(),
            html.Br(),
                        
            html.H2('''EVOLUTION OF AN INDICATOR'''
            , style={'textAlign':'center'}),
            
            html.Br(),

            html.Label(['Select Indicator:', dcc.Dropdown(placeholder='Indicator X:',
                id='indicator_polar',
                options=[{'label':x, 'value' : x} for x in indicators],
                value='CO2 emissions (kt per 1,000)')],
                style={'font-size':'11px', 'font-weight':'bold'}
            ),

            html.Br(),

            html.P('''By playing with the Bar Polar Chart, you will be able to activate an animation by clicking on play, and which
            will display the evolution of the indicator you have previously selected in the available data years from 2000 to 2020, structured into the
            different countries of the EU.'''),

            html.P('''You can interact with the Bar Polar Chart by pausing the animation at a given Year, or by hovering the cursor
            over it: you will se the Country Name, the Year and the rate of the Indicator selected
            ''')
            
        ],
        style={'height': '500px', 'width':'33%', 'marginRight':'1%', 'marginLeft':'1%', 'float':'right'} 
        ),

        html.P(children=[        

                    
            dcc.Graph(
                id='barpolarplot',
                figure=fig2,
                style={'width': '100%'}
            ),
        ], style={'width':'64%', 'marginLeft':'1%'})

    ],style={'width':'100%','height':'530px', 'backgroundColor':'lightsteelblue', 'marginTop':'1%'}, 
    ),

    html.Br(),

    html.Div(style={'height': '10px','width':'100%'}),

    html.Div(children=[

        html.P(children=[
                        
            dcc.Graph(
                id='scatterplot',
                figure=fig,
                style={'width': '100%'})
            
            ], style={'width':'64%', 'marginRight':'1%', 'float':'right'}),

        html.P(children=[ 

            html.Br(),

            html.Br(),

            html.H2('''EVOLUTION OF THE RELATIONSHIP BETWEEN TWO INDICATORS'''
            , style={'textAlign':'center'}),


            html.P(children=[
                html.Label(['Select Indicator X:', dcc.Dropdown(placeholder='Indicator X:',
                    options=[{'label':x, 'value' : x} for x in indicators],
                    id='Indicator_X',
                    value='Female labor force (% Labor)')],
                    style={'font-size':'11px','font-weight':'bold'}
                ),
                
                html.Label(['Select Indicator Y:', dcc.Dropdown(placeholder='Indicator Y:',
                    id='Indicator_y',
                    options=[{'label':x, 'value' : x} for x in indicators],
                    value='Female population (% Pop.)')],
                    style={'font-size':'11px', 'font-weight':'bold'}
                )
            ]),
            
            html.P('''By playing with the Scatter Plot, you will be able to activate an animation by clicking on play, and which
            will display the evolution of the relationship between the two indicators you have previously selected in the available data years from 2000 to 2020,
            structured into the different countries of the EU.'''),

            html.P('''You can interact with the Scatter plot by pausing the animation at a given Year, or by hovering the cursor
            over it: you will se the Country Name, the Year and the rate of the two Indicators selected''')
        ],
        style={'width':'33%', 'height': '500px', 'marginLeft':'1%', 'marginRight':'1%'} 
        )

        ], style={'width':'100%', 'backgroundColor':'lightsteelblue', 'height':'530px'}),

    html.Br(),

    html.Div(style={'height': '10px','width':'100%'}),

    html.Div(children=[

        html.P(style={'height':'0.3%'}),
        
        html.P(children=[

            html.Br(),
            html.Br(),
                        
            html.H2('''DIMENSIONALITY REDUCTION INDEX'''
            , style={'textAlign':'center'}),
            
            html.Br(),

            html.P('''By applying statistical techniques of dimensionailty reduction, I have reduced all the indicator to just one variable, thus creating an
            Index by how the countries of the European Union are coping with the SDG, not only in the present but also adding available historical data'''),

            html.P('''It is important to point out that this Index is not official and is based on the data obtained from the World Bank. The data has been 
            reduced and standarised to make it individual as an Index and to facilitate the comparision between countries. Just consists of
            an approach made to provide the user with an overview of all the tools listed before.
            ''')
            
        ],
        style={'height': '500px', 'width':'33%', 'marginRight':'1%', 'marginLeft':'1%', 'float':'right'} 
        ),

        html.P(children=[        

                    
            dcc.Graph(
                id='index_plot',
                figure=fig4,
                style={'width': '100%'}
            ),
        ], style={'width':'64%', 'marginLeft':'1%'})

    ],style={'width':'100%','height':'530px', 'backgroundColor':'lightsteelblue', 'marginTop':'1%'}, 
    ),

    html.Div(style={'height': '10px','width':'100%'})



], style={'font-family':'Arial', 'padding-left':'133px', 'padding-right':'133px', 'padding-top':'21px'})


@app.callback(
    Output('image_indicator', 'src'),
    [Input('radio_items', 'value')])
def change_picture(radio_items):
    im_indicator=app.get_asset_url(f'{radio_items}.jpg')

    return im_indicator

@app.callback(
    [Output('check_list_ind', 'options'),
    Output('check_list_ind', 'value')],
    [Input('radio_items', 'value')])
def change_radio(radio_items):
    options=[{'label':x, 'value' : x} for x in dictionary[radio_items]]
    value= [x for x in dictionary[radio_items]]
                
    return options, value

@app.callback(
    Output('Indicator', 'options'),
    [Input('Target', 'value')])
def set_indicator_options(Target):
    return [{'label': i, 'value': i} for i in dictionary[Target]]

@app.callback(
    [Output('Year', 'options'),
    Output('Year', 'value')],
    [Input('Indicator', 'value')])
def set_year_options(Indicator):

    value = list(-np.sort(-df2[df2[Indicator] != 0]['Year'].unique()))[0]

    return [{'label': i, 'value': i} for i in list(-np.sort(-df2[df2[Indicator] != 0]['Year'].unique()))], value

@app.callback(
    Output('interactive_map', 'figure'),
    [Input('Target', 'value'),
    Input('Indicator', 'value'),
    Input('Year', 'value')])
def edit_map(Target, Indicator, Year):

    new_df = df2[(df2['Year'] == Year) & (df2[Indicator] !=0)]
    new_df['Rank'] = new_df[Indicator].rank(ascending=False)
    new_df['Average_EU'] = round(new_df[Indicator].mean(),2)
    
    fig3=px.choropleth(new_df, geojson=geo_objects, featureidkey="properties.ISO3",locations='Country ISO3',
                      color_continuous_scale="Greens" if dictionary2[Indicator] == 'Positive' else 'Reds',
                      color=Indicator, scope='europe',
                     projection='natural earth',
                     template='ggplot2',height=500,
                     hover_name='Country Name',
                     hover_data={'Country ISO3':False, Indicator:True, 'Year':True, 'Average_EU':True, 'Rank':True})
    
    fig3.update_layout(font_family='Arial', font_size=10, hoverlabel_font_size=10,
    margin=dict(l=0, r=0, t=0, b=0), hoverlabel_bgcolor='Green' if dictionary2[Indicator] == 'Positive' else 'DarkRed')
    fig3.update_coloraxes(colorbar_len=0.7)
    fig3.update_coloraxes(colorbar_ticks="")
    fig3.update_coloraxes(colorbar_outlinewidth=0.21)
    fig3.update_coloraxes(colorbar_thickness=30)
    fig3.update_coloraxes(colorbar_title_text='Indicator Range')
    fig3.update_coloraxes(colorbar_title_side='right')
    fig3.update_coloraxes(colorbar_ticklabelposition='inside')
    
    return fig3


@app.callback(
    Output('barpolarplot', 'figure'),
    [Input('indicator_polar', 'value')])

def polar_plot(indicator_polar):

    fig2 = px.bar_polar(df2_gb, r=indicator_polar, theta="Country ISO3",
                       color=indicator_polar,
                       template="ggplot2",
                       color_continuous_scale= px.colors.sequential.Greens if dictionary2[indicator_polar] == 'Positive' else px.colors.sequential.Reds,
                       range_color=[df2_gb[indicator_polar].min(),df2_gb[indicator_polar].max()],
                       range_r=[df2_gb[indicator_polar].min(),df2_gb[indicator_polar].max()],
                       hover_name='Country Name',
                       hover_data={'Year':True, indicator_polar:':.2f', 'Country ISO3':False},
                       animation_frame='Year',
                       height=500)
    
    fig2.update_layout(font_family='Arial', font_size=10, hoverlabel_font_size=10)
    fig2.update_coloraxes(colorbar_len=0.7)
    fig2.update_coloraxes(colorbar_ticks="")
    fig2.update_polars(radialaxis_visible=False)
    fig2.update_coloraxes(colorbar_outlinewidth=0.21)
    fig2.update_coloraxes(colorbar_thickness=30)
    fig2.update_coloraxes(colorbar_title_text='Indicator Range')
    fig2.update_coloraxes(colorbar_title_side='right')
    fig2.update_coloraxes(colorbar_ticklabelposition='inside')
    
    return fig2

@app.callback(
    Output('scatterplot', 'figure'),
    [Input('Indicator_X', 'value'),
    Input('Indicator_y', 'value')])

def scatter_display(Indicator_X,
                    Indicator_y,):
    
    fig = px.scatter(df2_gb, x=Indicator_X, y=Indicator_y, 
               animation_frame="Year",
               category_orders={'Country Name': countries, 'Year': years},
               hover_name="Country Name",
               color="Country Name",
               range_y=[df2_gb[Indicator_y].min()-10,df2_gb[Indicator_y].max()+10] if Indicator_y != 'GDP per capita (current US$)' else [df2_gb[Indicator_y].min() - 500,df2_gb[Indicator_y].max()+ 1000],
               log_y=True if Indicator_y == 'GDP per capita (current US$)' else False,
               size='Size',
               text='Country Name',
               opacity=0.33,
               hover_data={'Country Name':False, 'Year':True, 'Size':False, Indicator_X:':.2f', Indicator_y:':.2f'},
               range_x=[df2_gb[Indicator_X].min()-10,df2_gb[Indicator_X].max()+10] if Indicator_X != 'GDP per capita (current US$)' else [df2_gb[Indicator_X].min() - 500,df2_gb[Indicator_X].max()+ 1000],
               template='ggplot2',
               height=500)
    
    fig.update_layout(transition = {'duration': 20}, font_family='Arial', font_size=10, hoverlabel_font_size=10)
       
    return fig



if __name__ == '__main__':
    app.run_server(debug=True)

