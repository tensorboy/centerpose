import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import math

sizeref = 2000

# Dictionary with dataframes for each continent
continent_names = ['DLA', 'Resnet', 'MobileNet', 'ShuffleNet', 'HigherResolution', 'HardNet']
continent_data = {}

continent_data['DLA-34'] = {'map':[62.3], 'speed':[23], 'size':82.7/2}
continent_data['Resnet50'] = {'map':[53.0], 'speed':[28], 'size':139.8/2}
continent_data['MobileNetV3'] = {'map':[45.1], 'speed':[30], 'size':9.7/2}
continent_data['ShuffleNetV2'] = {'map':[37.7], 'speed':[25], 'size':40./2}
continent_data['HigherResolution'] = {'map':[49.5], 'speed':[16], 'size':115.2/2}
continent_data['HardNet'] = {'map':[41.9], 'speed':[30], 'size':19.3/2}


# Create figure
fig = go.Figure()

for continent_name, continent in continent_data.items():
    fig.add_trace(go.Scatter(
        x=continent['speed'], y=continent['map'],
        name=continent_name, text='model performance',
        marker_size=continent['size'],
        ))
# Tune marker appearance and layout
fig.update_traces(mode='markers', marker=dict(sizemode='area',
                                              sizeref=sizeref, line_width=2))

fig.update_layout(
    title='mAP v.s. Runtime',
    xaxis=dict(
        title='Run Time (ms)',
        gridcolor='white',
        type='log',
        gridwidth=2,
    ),
    yaxis=dict(
        title='Mean Average Precision (mAP)',
        gridcolor='white',
        gridwidth=2,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)
fig.show()
