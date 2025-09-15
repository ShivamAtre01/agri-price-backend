import pandas as pd
import numpy as np
from datetime import datetime
import calendar
import os

def preprocess_data():
    """
    Preprocess onion price and rainfall data:
    1. Keep rainfall data at monthly level
    2. Merge datasets based on year and month
    3. Handle missing values
    """
    print("Starting data preprocessing...")
    
    # Load datasets
    onion_df = pd.read_csv('onion.csv')
    rainfall_df = pd.read_csv('Rainfall.csv')
    
    # Convert date format in onion dataset
    onion_df['date'] = pd.to_datetime(onion_df['date'], format='%d-%m-%Y')
    onion_df['year'] = onion_df['date'].dt.year
    onion_df['month'] = onion_df['date'].dt.month
    onion_df['day'] = onion_df['date'].dt.day
    
    print(f"Onion dataset shape: {onion_df.shape}")
    print(f"Onion dataset date range: {onion_df['date'].min()} to {onion_df['date'].max()}")
    print(f"Onion dataset years: {onion_df['year'].min()} to {onion_df['year'].max()}")
    
    # Print rainfall dataset years
    print(f"Rainfall dataset years: {rainfall_df['YEAR'].min()} to {rainfall_df['YEAR'].max()}")
    
    # Reshape rainfall data from wide to long format (monthly)
    month_map = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    
    # Create a list to store the reshaped data
    rainfall_monthly = []
    
    for _, row in rainfall_df.iterrows():
        year = row['YEAR']
        for month_name, month_num in month_map.items():
            rainfall_monthly.append({
                'year': year,
                'month': month_num,
                'rainfall': row[month_name]
            })
    
    # Convert to DataFrame
    rainfall_monthly_df = pd.DataFrame(rainfall_monthly)
    
    print(f"Reshaped rainfall dataset shape: {rainfall_monthly_df.shape}")
    
    # Check if we have rainfall data for all the years we need
    min_year = onion_df['year'].min()
    max_year = onion_df['year'].max()
    
    available_years = rainfall_monthly_df['year'].unique()
    required_years = list(range(min_year, max_year + 1))
    missing_years = [year for year in required_years if year not in available_years]
    
    if missing_years:
        print(f"Warning: Missing rainfall data for years: {missing_years}")
        print("Will use data from available years and apply seasonal patterns")
        
        # If we have missing years, we'll use the average rainfall pattern from available years
        avg_monthly_rainfall = rainfall_monthly_df.groupby('month')['rainfall'].mean().reset_index()
        
        # Create entries for missing years using the average pattern
        for year in missing_years:
            for _, row in avg_monthly_rainfall.iterrows():
                rasinfall_monthly_df = pd.concat([
                    rainfall_monthly_df, 
                    pd.DataFrame([{
                        'year': year,
                        'month': row['month'],
                        'rainfall': row['rainfall']
                    }])
                ], ignore_index=True)
    
    # Filter to the years we need
    rainfall_monthly_df = rainfall_monthly_df[
        (rainfall_monthly_df['year'] >= min_year) & 
        (rainfall_monthly_df['year'] <= max_year)
    ]
    
    print(f"Filtered rainfall dataset shape: {rainfall_monthly_df.shape}")
    
    # Merge onion and rainfall datasets on year and month
    # First, aggregate onion prices to monthly level (average price per month)
    onion_monthly = onion_df.groupby(['year', 'month'])['value'].mean().reset_index()
    onion_monthly.rename(columns={'value': 'monthly_avg_price'}, inplace=True)
    
    # Merge the monthly datasets
    merged_monthly = pd.merge(onion_monthly, rainfall_monthly_df, on=['year', 'month'], how='left')
    
    # Check for missing values
    missing_rainfall = merged_monthly['rainfall'].isna().sum()
    print(f"Missing rainfall values after merge: {missing_rainfall}")
    
    # Fill missing rainfall values with monthly averages across years
    if missing_rainfall > 0:
        # Group by month to get monthly averages across years
        monthly_avg = merged_monthly.groupby('month')['rainfall'].transform('mean')
        merged_monthly.loc[merged_monthly['rainfall'].isna(), 'rainfall'] = monthly_avg
        
        # If there are still missing values, fill with overall average
        still_missing = merged_monthly['rainfall'].isna().sum()
        if still_missing > 0:
            print(f"Still missing rainfall values after monthly averaging: {still_missing}")
            overall_avg = merged_monthly['rainfall'].mean()
            merged_monthly.loc[merged_monthly['rainfall'].isna(), 'rainfall'] = overall_avg
    
    # Now merge the monthly rainfall data back to the daily onion price data
    final_df = pd.merge(onion_df, merged_monthly[['year', 'month', 'rainfall', 'monthly_avg_price']], 
                        on=['year', 'month'], how='left')
    
    # Add some features that might be useful for time series forecasting
    final_df['day_of_week'] = final_df['date'].dt.dayofweek
    final_df['day_of_year'] = final_df['date'].dt.dayofyear
    final_df['quarter'] = final_df['date'].dt.quarter
    
    # Calculate price difference from monthly average
    final_df['price_diff_from_avg'] = final_df['value'] - final_df['monthly_avg_price']
    
    # Add lag features (previous day, previous week, previous month)
    final_df['prev_day_price'] = final_df['value'].shift(1)
    final_df['prev_week_price'] = final_df['value'].shift(7)
    final_df['prev_month_price'] = final_df['value'].shift(30)
    
    # Add rolling averages
    final_df['rolling_7d_avg'] = final_df['value'].rolling(window=7, min_periods=1).mean()
    final_df['rolling_30d_avg'] = final_df['value'].rolling(window=30, min_periods=1).mean()
    
    # Fill NaN values created by shifts and rolling calculations
    for col in ['prev_day_price', 'prev_week_price', 'prev_month_price', 
                'rolling_7d_avg', 'rolling_30d_avg']:
        final_df[col].fillna(method='bfill', inplace=True)
    
    # Save the preprocessed dataset
    final_df.to_csv('preprocessed_data.csv', index=False)
    
    print("Data preprocessing completed. Saved as 'preprocessed_data.csv'")
    print(f"Final dataset shape: {final_df.shape}")
    
    # Print some statistics about the final dataset
    print("\nStatistics for onion prices:")
    print(final_df['value'].describe())
    
    print("\nStatistics for rainfall:")
    print(final_df['rainfall'].describe())
    
    return final_df

if __name__ == "__main__":
    preprocess_data()
