from itertools import chain
import os
import numpy as np
import plotly.graph_objects as go
import pandas as pd



model_list = []
val_list = []
epoch_list = []


def get_list(dirs,model_list, val_list, epoch_list):
    for directories in os.listdir(dirs):
        model_list.append(directories)
        model_name = directories.split("_")[0]
        file_dir = dirs + directories
        for file in os.listdir(file_dir):
            epoch_list.append(file.split("-")[1])
            val_list.append(file.split("-")[2])
    epoch_array = np.array(epoch_list).reshape(-1, 3)
    val_array = np.array(val_list).reshape(-1, 3)
    return val_array, epoch_array


def array_tovalue(array):
    values = array.tolist()
    values_temp = list(chain(*values))
    return np.float_(values_temp)


def add_bar(df,fig, val_array, epoch_array):
    x_values = array_tovalue(epoch_array)
    y_values = array_tovalue(val_array)
    for i in range(int(len(model_list))):
        fig.add_trace(go.Bar(x=x_values[i * epoch_array.shape[1]:(i + 1) * epoch_array.shape[1]],
                             y=y_values[i * val_array.shape[1]:(i + 1) * val_array.shape[1]],
                             name=np.str_(model_list[i]),
                             width=0.4))
        fig.update_yaxes(range=[0.7, 0.9])

    fig.add_trace(go.Bar(x=df["type"], y=df["acc"], name="ensemble", width=0.37))
    return fig


def main():
    number=input("MX_800_5-")
    dirs = "../dataset/MX_800_5-"+number+"/model_save/"
    dir_ensemble = "../dataset/MX_800_5-"+number+"/train_result/ensemble_result.csv"
    df = pd.read_csv(dir_ensemble)

    val_array, epoch_array = get_list(dirs,model_list, val_list, epoch_list)

    fig = go.Figure()
    fig = add_bar(df,fig, val_array, epoch_array)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue",
        title="Validation Accuracy by Models",
        xaxis_type="category",
        xaxis_title="epoch",
        yaxis_title="Validation Accuracy",
        yaxis=dict(categoryorder='category ascending'),
        # showlegend=False, 주석 유무
    )
    fig.show()


if __name__ == '__main__':
    main()
