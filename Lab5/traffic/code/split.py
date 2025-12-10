def series_train_test_split(data, trn_pct: float = 0.90):
    trn_size = int(len(data) * trn_pct)
    df_cp = data.copy()
    train = df_cp.iloc[:trn_size, 0]
    test = df_cp.iloc[trn_size:, 0]
    return train, test

def split_data(
    df,
    target_column: str = 'Total', 
    train_size: float = 0.3, 
):
    """
    Splits the dataset into training and testing sets.
    """
    print(f"--- Splitting data with train size = {train_size} ---")

    timeseries = df[target_column]
    
    train, test = series_train_test_split(timeseries, trn_pct=train_size)
    
    print(f"Data split completed: {len(train)} training samples and {len(test)} testing samples.")
    
    return train, test