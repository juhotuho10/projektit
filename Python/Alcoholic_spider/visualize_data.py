import os
from datetime import datetime

import pandas as pd
import plotly.express as px


def str_to_time(str_time: str):
    strings = str_time.rstrip(".csv").split("_")
    month, year = int(strings[-2]), int(strings[-1])

    return pd.to_datetime(datetime(year=year, month=month, day=1))


def load_csv(dir_path):
    dataframes = pd.DataFrame()
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(".csv"):
            full_path = os.path.join(dir_path, filename)
            df = pd.read_csv(full_path)

            df = df[["Name", "Alcohol_per_euro_per_liter"]]
            filt = (~df["Alcohol_per_euro_per_liter"].isna()) & (df["Alcohol_per_euro_per_liter"] != 0)
            df = df[filt]
            df["Alcohol_per_euro_per_liter"] = df["Alcohol_per_euro_per_liter"].astype(float)
            df["date_time"] = str_to_time(filename)
            dataframes = pd.concat([dataframes, df])

    return dataframes


def plot_df(df):
    fig = px.line(
        df,
        x="date_time",
        y="Alcohol_per_euro_per_liter",
        color="Name",
        hover_name="Name",
        hover_data={"date_time": False},
        title="Alcohol per Euro per Liter Over Time",
    )

    fig.update_traces(mode="markers+lines", marker=dict(size=6))
    fig.update_layout(xaxis_title="Date", yaxis_title="Alcohol per € per Litre", hovermode="closest")

    fig.show()


if __name__ == "__main__":
    folder = "./Price_data"
    collected_df = load_csv(folder)

    alcohol_filt = collected_df["Alcohol_per_euro_per_liter"] > 1.3
    most_alcohol_df = collected_df[alcohol_filt]

    plot_df(most_alcohol_df)
