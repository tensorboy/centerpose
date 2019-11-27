import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import math

# Load data, define hover text and bubble size
data = px.data.gapminder()
df_2007 = data[data['year']==2007]
df_2007 = df_2007.sort_values(['continent', 'country'])

hover_text = []
bubble_size = []

for index, row in df_2007.iterrows():
    hover_text.append(('Country: {country}<br>'+
                      'Life Expectancy: {lifeExp}<br>'+
                      'GDP per capita: {gdp}<br>'+
                      'Population: {pop}<br>'+
                      'Year: {year}').format(country=row['country'],
                                            lifeExp=row['lifeExp'],
                                            gdp=row['gdpPercap'],
                                            pop=row['pop'],
                                            year=row['year']))
    bubble_size.append(math.sqrt(row['pop']))

df_2007['text'] = hover_text
df_2007['size'] = bubble_size
sizeref = 2000

# Dictionary with dataframes for each continent
continent_names = ['DLA', 'Resnet', 'MobileNet', 'ShuffleNet', 'HigherResolution', 'HardNet']
continent_data = {}

continent_data['DLA'] = {'map':[62.3], 'speed':[23]}
continent_data['Resnet'] = {'map':[53.0], 'speed':[28]}
continent_data['MobileNet'] = {'map':[45.1], 'speed':[30]}
continent_data['ShuffleNet'] = {'map':[33.3], 'speed':[25]}
continent_data['HigherResolution'] = {'map':[45.2], 'speed':[16]}
continent_data['HardNet'] = {'map':[30.], 'speed':[30]}


# Create figure
fig = go.Figure()

for continent_name, continent in continent_data.items():
    fig.add_trace(go.Scatter(
        x=continent['speed'], y=continent['map'],
        name=continent_name, text='model performance',
        marker_size=40,
        ))

# Tune marker appearance and layout
fig.update_traces(mode='markers', marker=dict(sizemode='area',
                                              sizeref=sizeref, line_width=2))

fig.update_layout(
    title='mAP v.s. Runtime',
    xaxis=dict(
        title='Runtime (ms)',
        gridcolor='white',
        type='log',
        gridwidth=2,
    ),
    yaxis=dict(
        title='mean average precision (mAP)',
        gridcolor='white',
        gridwidth=2,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)
fig.show()
