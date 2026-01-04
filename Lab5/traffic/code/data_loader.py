import pandas as pd

def load_data(file_path='Datasets/TrafficTwoMonth.csv'):
    df = pd.read_csv(file_path)

    ## Data for TWO months
    df['Month'] = None
    df.loc[:2111, 'Month'] = '2023-12-'
    df.loc[2112:5087, 'Month'] = '2024-01-'
    df.loc[5088:, 'Month'] = '2024-02-'

    df['Datetime_Str'] = (
        df['Month'] + 
        df['Date'].astype(str).str.zfill(2) + 
        ' ' + 
        df['Time'].astype(str)
    )

    df['Datetime'] = pd.to_datetime(df['Datetime_Str'], format='%Y-%m-%d %I:%M:%S %p')
    df.drop(columns=['Month'], inplace=True)

    df = df.set_index('Datetime').sort_index()
    df.drop(columns=['Datetime_Str'], inplace=True)
    
    return df