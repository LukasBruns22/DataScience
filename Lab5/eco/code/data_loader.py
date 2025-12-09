import pandas as pd

def load_data(file_path='Datasets/economic_indicators_dataset_2010_2023.csv'):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Country'] == 'USA']
    df = df.drop(columns=['Country'])

    df = df.groupby('Date').mean()  # Aggregate by Date if multiple entries exist

    df = df.sort_index()
    
    return df