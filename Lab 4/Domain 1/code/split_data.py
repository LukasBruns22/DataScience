import pandas as pd

df = pd.read_csv('datasets/domain1_final.csv')

# Filter
train_df = df[df['Set'] == 'Train'].drop(columns=['Set'])
test_df = df[df['Set'] == 'Test'].drop(columns=['Set'])

# Save
train_df.to_csv('datasets/traffic_accidents_prepared_train.csv', index=False)
test_df.to_csv('datasets/traffic_accidents_prepared_test.csv', index=False)