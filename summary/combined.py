import pandas as pd
import os

def combine_csvs(folder='.', output_file='combined.csv'):
    all_dfs = []

    for file in os.listdir(folder):
        if file.endswith('.csv'):
            path = os.path.join(folder, file)
            df = pd.read_csv(path)
            df.insert(0, 'Source', os.path.splitext(file)[0])
            all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(os.path.join(folder, output_file), index=False)

if __name__ == '__main__':
    combine_csvs()
