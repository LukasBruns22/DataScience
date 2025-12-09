import pandas as pd

def load_data(file_path='Datasets/TrafficTwoMonth.csv'):
    df = pd.read_csv(file_path)

    df['Datetime_Str'] = (
        '2023-12-' + 
        df['Date'].astype(str).str.zfill(2) + 
        ' ' + 
        df['Time'].astype(str)
    )

    df['Datetime'] = pd.to_datetime(df['Datetime_Str'], format='%Y-%m-%d %I:%M:%S %p')

    df = df.set_index('Datetime').sort_index()
    df.drop(columns=['Datetime_Str'], inplace=True)
    
    return df