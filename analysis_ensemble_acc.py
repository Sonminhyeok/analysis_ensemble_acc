from itertools import chain

import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go

dir = "../dataset/MX_800_5-01/model_save/"
model_list = []
val_list = []
epoch_list = []
for directories in os.listdir(dir):
    model_list.append(directories)
    model_name = directories.split("_")[0]
    file_dir = dir + directories
    for file in os.listdir(file_dir):
        epoch_list.append(file.split("-")[1])
        val_list.append(file.split("-")[2])
epoch_array = np.array(epoch_list).reshape(-1, 3)
val_array = np.array(val_list).reshape(-1, 3)

fig = go.Figure()
for i in range(int(len(val_list)/len(model_list))):
    y_values = val_array[:,i:i+1].tolist()
    y_values_temp=list(chain(*y_values))
    y_values=y_values_temp
    fig.add_trace(go.Bar(x=model_list,y=y_values))
    # fig.update_yaxes(range=[-1,5])
fig.update_layout(

    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",
    title="Validation Accuracy by Models",
    xaxis_title="Epoch",
    yaxis_title="Validation Accuracy",
    yaxis=dict(range=[0,100],categoryorder='category ascending')
)

fig.show()
