import pandas as pd
import os
import argparse

def clean_data(input_file, output_file, target_var):
    """
    Load data, clean and save the cleaned dataset.
    """
    if not os.path.exists(input_file):
        print(f"Error: File not found at {input_file}")
    else:
        df = pd.read_csv(input_file)
        print(f"Original Shape: {df.shape}")

        # 2. Drop variables (columns) that are totally empty
        df = df.dropna(axis=1, how='all')

        # 3. Drop records (rows) having ANY missing values
        df = df.dropna(axis=0, how='any')

        # 4. Encode the Target 
        # We convert 'Cancelled' to numbers (e.g., Yes=1, No=0) so it survives the cleaning
        if target_var in df.columns:
            df[target_var] = df[target_var].astype('category').cat.codes
            print(f"Target '{target_var}' encoded to numeric.")
        else:
            print(f"⚠️ WARNING: Target variable '{target_var}' not found in dataset!")

        # 5. Discard all non-numeric data
        # Now that 'Cancelled' is a number, it will be kept.
        df_numeric = df.select_dtypes(include=['number'])

        # 6. Save the clean file
        df_numeric.to_csv(output_file, index=False)

        # Final Check
        print("-" * 30)
        print("Processing complete.")
        print(f"Cleaned Shape:  {df_numeric.shape}")
        print(f"Columns kept:   {df_numeric.columns.tolist()}")
        print(f"File saved to:  {output_file}")

if __name__ == "__main__":
    # --- DÉFINITION DES ARGUMENTS ---
    parser = argparse.ArgumentParser(
        description="Clean a CSV dataset by removing empty rows/columns, encoding the target variable, and keeping only numeric data."
    )
    
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        required=True, 
        help='Path to the input CSV file to be cleaned (e.g., datasets/file.csv)'
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        required=True, 
        help='Path to save the cleaned CSV file (e.g., datasets/file_cleaned.csv)'
    )
    parser.add_argument(
        '-t', '--target', 
        type=str, 
        required=True, 
        help='Name of the target variable/column to encode (e.g., Cancelled)'
    )

    args = parser.parse_args()

    clean_data(args.input, args.output, args.target)