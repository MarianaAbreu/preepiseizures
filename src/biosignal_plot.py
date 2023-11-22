# Script with plotting functions for biosignals
#
# created by: Mariana Abreu
# date: 29 August 2023
#
# built-in
import os

# external
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# local

def plot_acc_seizure(data, seizure_row, plotname):
    """
    Plot ACC data for a seizure
    Parameters:
        data (dict): dictionary with patient data
        seizure_row (): row of seizure table
    Returns:
        Save plot
    """
    # get seizure info columns
    seiz_info_cols = [col for col in seizure_row.index if col in ['Focal / Generalisada', 'Tipo.1']]
    # create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, specs=[[{"secondary_y": True}], [{"secondary_y": True}]])
    # get chest activity index data
    chest_acc = data[['activity_index_chestbit', 'datetime_chestbit', 'acc_magnitude_chestbit']].dropna()
    # get wrist activity index data
    wrist_acc = data[['activity_index_wristbit', 'datetime_wristbit', 'acc_magnitude_wristbit']].dropna()
    # add traces
    fig.add_trace(go.Scatter(x=chest_acc['datetime_chestbit'], y=chest_acc['activity_index_chestbit'], name='Activity Index Chest',
                          line=dict(color='#C76A3E', width=4)), row=1, col=1)
    fig.add_vrect(
    x0=str(seizure_row['Timestamp']),
    x1=str(seizure_row['Timestamp'] + pd.Timedelta(seconds=10)),
    label=dict(
        text='Onset',
        textposition="top right",
        font=dict(size=20, family="Times New Roman"),
    ),
    fillcolor="green",
    opacity=0.9,
    line_width=0,row=1, col=1)

    fig.add_vrect(
    x0=str(seizure_row['Timestamp']+ pd.Timedelta(seconds=5)),
    x1=str(seizure_row['Timestamp'] + pd.Timedelta(seconds=121)),
    fillcolor="green",
    opacity=0.25,
    line_width=0,row=1, col=1)
    
    fig.add_trace(go.Scatter(x=wrist_acc['datetime_wristbit'], y=wrist_acc['activity_index_wristbit'], name='Activity Index Wrist', 
                          line=dict(color='#4D6FAC', width=4)), row=2, col=1)

    fig.add_vrect(
    x0=str(seizure_row['Timestamp']),
    x1=str(seizure_row['Timestamp'] + pd.Timedelta(seconds=10)),
    label=dict(
        text='Onset',
        textposition="top right",
        font=dict(size=20, family="Times New Roman"),
    ),
    fillcolor="green",
    opacity=0.9,
    line_width=0,row=2, col=1)

    fig.add_vrect(
    x0=str(seizure_row['Timestamp']+ pd.Timedelta(seconds=5)),
    x1=str(seizure_row['Timestamp'] + pd.Timedelta(seconds=121)),
    fillcolor="green",
    opacity=0.25,
    line_width=0,row=2, col=1)
    fig.update_layout(title=str(seizure_row[seiz_info_cols].values), title_x=0.5, title_font_size=30, title_font_family="Times New Roman")
    fig.update_yaxes(title_text="Chest Activity Index", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Wrist Activity Index", row=2, col=1, secondary_y=False)
    if not plotname:
        plotname = 'acc_seizure_' + str(data['seizure'].unique())
    fig.write_image(f"/Users/saraiva/dev/PreEpiSeizuresCode/data/figures/{plotname}.png")
