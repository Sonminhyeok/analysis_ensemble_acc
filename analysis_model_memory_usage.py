import pandas as pd
import re
import plotly.graph_objs as go
import matplotlib.pyplot as plt

column1 = ["user", "_pid", "pid", "cpu_percent", "start_time", "duration", "command"]
column2 = ["_1", "_2", "_3", "_4", "pid", "_6", "cpu_usage"]
dirs = "../dataset/tr_MX_800_5_14p_DF2"
pid_dir = dirs + "/pid_log_tr_MX_800_5_14p_DF2.txt"
nvidia_dir = dirs + "/nvidia_log_tr_MX_800_5_14p_DF2.txt"
pid_csv_file_path = dirs + "/pid_log_tr_MX_800_5_14p_DF2" + ".csv"
nvidia_csv_file_path = dirs + "/nvidia_log_tr_MX_800_5_14p_DF2" + ".csv"


def text_to_df(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    data = [line.strip().split(maxsplit=6) for line in lines]
    df = pd.DataFrame(data, columns=column1)
    return df


def select_model(model, df):
    model_df = df[df["command"].str.contains(model, case=True, regex=True)]
    return model_df


def return_usage(pid_list, df):
    usage_list = df["cpu_usage"].where(df["pid"].isin(pid_list))
    usage_list = usage_list.dropna()

    return usage_list.to_list()


def return_numbers(list):
    numbers_only = [int(re.sub(r'\D', '', s)) for s in list]
    return numbers_only


def main():
    """
    txt to csv
    df = text_to_df(nvidia_dir)
    df.to_csv(nvidia_csv_file_path, index=False)

    """

    while True:
        input_number = input("(00~13) : ")
        if input_number == "exit":
            break
        Df = "MX_800_5-" + input_number
        input_model = input("(atn, cnn, ltm, tsf, mlp) : ")
        models = input_model

        df_pid = pd.read_csv(pid_csv_file_path)
        df_nvidia = pd.read_csv(nvidia_csv_file_path)

        temp = select_model(models, df_pid)
        df_pid = temp
        parsed_pid = df_pid["_pid"].where(df_pid["command"].str.contains(Df, case=True, regex=True)).unique()
        usage_list = return_usage(parsed_pid, df_nvidia)
        temp = return_numbers(usage_list)
        usage_list = temp
        """
        #matplotlib 사용시 코드
        plt.plot(usage_list)
        plt.title(Df+" ("+models+")")
        plt.xlabel('')
        plt.ylabel('memory_usage')
        plt.show()
        """

        trace = go.Scatter(x=list(range(len(usage_list))), y=usage_list, mode='lines+markers',
                           line=dict(color="#32Cd32"), marker=dict(color='#FF0000', size=2))
        layout = go.Layout(
            title=Df + " (" + models + ")",
            xaxis=dict(title=''),
            yaxis=dict(title='memory_usage')
        )
        fig = go.Figure(data=[trace], layout=layout)
        fig.show()


if __name__ == '__main__':
    main()
