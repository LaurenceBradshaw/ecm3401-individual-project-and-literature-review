import pandas as pd
import os
from data_file_iter import Data_file_iterator

# Iterate through all device_gnss.csv files in the training dataset
# and compute the maximum value for each numeric column across the dataset
# Then save the results to dataset_stats.csv with columns: column_name, max_value

# These values will be used later for normalisation during model training

if __name__ == "__main__":
    base_path = "./smartphone-decimeter-2023/sdc2023"
    output_file = f"{base_path}/dataset_stats.csv"
    stats = {}
    data_iter = Data_file_iterator(base_path, split='train')
    mp_counter = 0

    for df in data_iter:
        for column in df.select_dtypes(include=['number']).columns:
            if column == 'Cn0DbHz':
                # linear scale for Cn0DbHz (column is in dB-Hz which is logarithmic and we want linear for input to NN) 
                df[column] = 10 ** (df[column] / 10)

            max_value = abs(df[column].max())
            if column not in stats or max_value > stats[column]:
                if column == 'Cn0DbHz':
                    stats[f"{column}_linear"] = max_value
                else:
                    stats[column] = max_value

        print(f"Processed file: {data_iter.get_current_file()}")

    if os.path.exists(output_file):
        os.remove(output_file)

    stats_df = pd.DataFrame(list(stats.items()), columns=['column_name', 'abs_max_value'])
    stats_df.to_csv(output_file, index=False)

    print("Dataset statistics saved to dataset_stats.csv")
    print(stats_df)
