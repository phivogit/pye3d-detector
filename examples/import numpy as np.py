import json
import numpy as np
import pandas as pd
from scipy.stats import zscore

def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return pd.DataFrame(data)

def detect_abrupt_changes(df, column_name, threshold):
    try:
        data = np.array([np.array(x) for x in df[column_name]])
        gradients = [np.abs(np.gradient(data[:, i])) for i in range(data.shape[1])]
        max_gradient = np.max(gradients, axis=0)
        return max_gradient > threshold
    except:
        gradient = np.abs(np.gradient(df[column_name].to_numpy()))
        return gradient > threshold

def detect_outliers_zscore(df, column_name, threshold=5):
    try:
        data = np.array([np.array(x) for x in df[column_name]])
        outliers = np.zeros(len(df), dtype=bool)
        for i in range(data.shape[1]):
            z = zscore(data[:, i])
            outliers |= np.abs(z) > threshold
        return outliers
    except:
        z = zscore(df[column_name].to_numpy())
        return np.abs(z) > threshold

def filter_data(df):
    abrupt_marker = detect_abrupt_changes(df, 'marker_position', threshold=3.3)
    outlier_sphere = detect_outliers_zscore(df, 'sphere_center', threshold=3)
    
    df['abrupt_marker'] = abrupt_marker
    df['outlier_sphere'] = outlier_sphere
    
    # Loại bỏ tất cả bản ghi có abrupt_marker hoặc outlier_sphere
    df_filtered = df[~(abrupt_marker | outlier_sphere)].copy()
    
    print("\nFiltering Statistics:")
    print(f"Abrupt marker movements: {abrupt_marker.sum()} records")
    print(f"Sphere center outliers: {outlier_sphere.sum()} records")
    print(f"Filtered {len(df) - len(df_filtered)} out of {len(df)} records ({(1 - len(df_filtered)/len(df))*100:.1f}%)")
    
    return df_filtered

def save_filtered_data(filtered_df, output_filepath):
    filtered_df.drop(columns=['abrupt_marker', 'outlier_sphere'], inplace=True, errors='ignore')
    filtered_df.to_json(output_filepath, orient='records', indent=4)
    print(f"\nFiltered data saved to {output_filepath}")

def main(filepath='eye_tracking_data3.json', output_filepath='eye_tracking_data3s.json'):
    df = load_data(filepath)
    filtered_df = filter_data(df)
    save_filtered_data(filtered_df, output_filepath)
    return filtered_df

if __name__ == "__main__":
    main()