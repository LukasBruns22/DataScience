import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_timeseries(train, test):

    transf = StandardScaler().fit(train_scaled)

    train_scaled = transf.transform(train)
    test_scaled = transf.transform(test)

    train_scaled_df = pd.DataFrame(train_scaled, index=train.index, columns=train.columns)
    test_scaled_df = pd.DataFrame(test_scaled, index=test.index, columns=test.columns)

    return train_scaled_df, test_scaled_df