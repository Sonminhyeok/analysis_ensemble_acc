from itertools import chain

import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import pandas as pd
dir = "../dataset/MX_800_5-01/model_save/"
dir_ensemble="../dataset/MX_800_5-01/train_result/ensemble_result.csv"
df=pd.read_csv(dir_ensemble)

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
print(epoch_array)

for i in range(int(len(val_list)/len(model_list))):

    y_values = val_array[:,i:i+1].tolist()
    y_values_temp=list(chain(*y_values))
    y_values=np.float_(y_values_temp)
    fig.add_trace(go.Bar(x=model_list,y=y_values,name=np.str_(epoch_list[])))
    fig.update_yaxes(range=[0.75,0.9])
fig.add_trace(go.Bar(x=df["type"],y=df["acc"],name="ensemble"))
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",
    title="Validation Accuracy by Models",
    xaxis_title="Model",
    yaxis_title="Validation Accuracy",
    yaxis=dict(categoryorder='category ascending'),
    showlegend=False,
    bargap=0.01
)

fig.show()
