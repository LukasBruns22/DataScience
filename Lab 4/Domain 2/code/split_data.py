import pandas as pd

df = pd.read_csv('datasets/combined_flights_prepared.csv')

# Filter
train_df = df[df['Set'] == 'Train'].drop(columns=['Set'])
test_df = df[df['Set'] == 'Test'].drop(columns=['Set'])

# Save
train_df.to_csv('datasets/corpus_train_prepared.csv', index=False)
test_df.to_csv('datasets/corpus_test_prepared.csv', index=False)