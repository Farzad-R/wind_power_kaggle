import pandas as pd


def convert_date_to_timestamp(df):
    """
    Args:
    a dataframe with a column name date. THe column has seconds format.
    Returns: a dataframe with index of timestamp in the form of yyy/mm/dd hh/mm/ss
    """
    df["timestamp"] = df["date"].apply(
        lambda x: pd.to_datetime(str(x), format="%Y%m%d%H")
    )
    df = df.set_index("timestamp")
    df = df.drop(columns=["date"])
    df = df.sort_index()
    return df
