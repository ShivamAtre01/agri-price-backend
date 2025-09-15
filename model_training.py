import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
import datetime
import joblib
import os
import pickle

def load_data():
    """Load and prepare the preprocessed data"""
    print("Loading preprocessed data...")
    df = pd.read_csv('preprocessed_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

def train_test_split(df, test_size=0.2):
    """Split the data into training and testing sets"""
    train_size = int(len(df) * (1 - test_size))
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    return train_data, test_data

def create_sequences(data, seq_length):
    """Create sequences for LSTM model"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values
        y = data.iloc[i + seq_length]['value']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_prophet_model(train_data):
    """Train a Prophet model on the training data"""
    print("Training Prophet model...")
    # Prepare data for Prophet
    prophet_data = train_data[['date', 'value']].rename(columns={'date': 'ds', 'value': 'y'})
    
    # Add rainfall as a regressor
    prophet_data['rainfall'] = train_data['rainfall'].values
    prophet_data['monthly_avg_price'] = train_data['monthly_avg_price'].values
    
    # Initialize and train the model
    model = Prophet(yearly_seasonality=True, 
                   weekly_seasonality=True, 
                   daily_seasonality=False,
                   changepoint_prior_scale=0.05)
    
    # Add regressors
    model.add_regressor('rainfall')
    model.add_regressor('monthly_avg_price')
    
    # Fit the model
    model.fit(prophet_data)
    
    return model

def train_lstm_model(train_data, seq_length=30):
    """Train an LSTM model on the training data"""
    print("Training LSTM model...")
    
    # Select features for LSTM - using all available features now
    features = ['value', 'rainfall', 'monthly_avg_price', 'month', 'day', 
                'day_of_week', 'day_of_year', 'quarter', 'price_diff_from_avg',
                'prev_day_price', 'prev_week_price', 'prev_month_price',
                'rolling_7d_avg', 'rolling_30d_avg']
    
    data = train_data[features]
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=features)
    
    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)
    
    # Reshape X for LSTM [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    
    # Build the LSTM model - making it deeper and with more units
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model with more epochs
    history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
    
    # Save the scaler and features for later use
    joblib.dump(scaler, 'lstm_scaler.pkl')
    with open('lstm_features.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    # plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('lstm_training_history.png')
    
    return model, scaler, features, history

def evaluate_prophet_model(model, test_data):
    """Evaluate the Prophet model on the test data"""
    print("Evaluating Prophet model...")
    
    # Prepare test data for Prophet
    future = test_data[['date', 'value', 'rainfall', 'monthly_avg_price']].rename(columns={'date': 'ds', 'value': 'y'})
    
    # Make predictions
    forecast = model.predict(future)
    
    # Extract actual and predicted values
    y_true = test_data['value'].values
    y_pred = forecast['yhat'].values
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    print(f"Prophet Model Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(test_data['date'], y_true, label='Actual')
    plt.plot(test_data['date'], y_pred, label='Prophet Prediction')
    plt.title('Prophet Model: Actual vs Predicted Onion Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('prophet_predictions.png')
    
    return forecast, rmse, mae, mape

def evaluate_lstm_model(model, scaler, features, test_data, seq_length=30):
    """Evaluate the LSTM model on the test data"""
    print("Evaluating LSTM model...")
    
    # Prepare the full dataset (we need some training data for the sequence)
    full_data = test_data[features].copy()
    
    # Scale the data
    scaled_full_data = scaler.transform(full_data)
    scaled_full_data = pd.DataFrame(scaled_full_data, columns=features)
    
    # Create sequences
    X, y_true = create_sequences(scaled_full_data, seq_length)
    
    # Reshape X for LSTM [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Flatten predictions if needed
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    print(f"LSTM Model Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    test_dates = test_data['date'].iloc[seq_length:seq_length+len(y_true)]
    plt.plot(test_dates, y_true, label='Actual')
    plt.plot(test_dates, y_pred, label='LSTM Prediction')
    plt.title('LSTM Model: Actual vs Predicted Onion Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('lstm_predictions.png')
    
    # Plot a zoomed-in view of the last 90 days
    if len(y_true) > 90:
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates[-90:], y_true[-90:], label='Actual')
        plt.plot(test_dates[-90:], y_pred[-90:], label='LSTM Prediction')
        plt.title('LSTM Model: Last 90 Days - Actual vs Predicted Onion Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig('lstm_predictions_last_90_days.png')
    
    return y_pred, rmse, mae, mape

def hybrid_predictions(prophet_preds, lstm_preds, weights=(0.3, 0.7)):
    """Combine predictions from both models using weighted average
    Default weights favor LSTM more since it typically performs better
    """
    print("Generating hybrid predictions...")
    
    # Ensure the predictions are the same length
    min_length = min(len(prophet_preds), len(lstm_preds))
    prophet_preds = prophet_preds[:min_length]
    lstm_preds = lstm_preds[:min_length]
    
    # Calculate weighted average
    hybrid_preds = weights[0] * prophet_preds + weights[1] * lstm_preds
    
    return hybrid_preds

def evaluate_hybrid_model(hybrid_preds, test_data, seq_length=30, lstm_pred_length=None):
    """Evaluate the hybrid model on the test data"""
    print("Evaluating hybrid model...")
    
    # Extract actual values - make sure we're using the same range as the hybrid predictions
    if lstm_pred_length is None:
        lstm_pred_length = len(hybrid_preds)
    
    y_true = test_data['value'].iloc[seq_length:seq_length+lstm_pred_length].values[:len(hybrid_preds)]
    
    # Ensure y_true and hybrid_preds have the same length
    min_length = min(len(y_true), len(hybrid_preds))
    y_true = y_true[:min_length]
    hybrid_preds = hybrid_preds[:min_length]
    
    print(f"Length of y_true: {len(y_true)}, Length of hybrid_preds: {len(hybrid_preds)}")
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, hybrid_preds))
    mae = mean_absolute_error(y_true, hybrid_preds)
    mape = mean_absolute_percentage_error(y_true, hybrid_preds) * 100
    
    print(f"Hybrid Model Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    test_dates = test_data['date'].iloc[seq_length:seq_length+min_length]
    plt.plot(test_dates, y_true, label='Actual')
    plt.plot(test_dates, hybrid_preds, label='Hybrid Prediction')
    plt.title('Hybrid Model: Actual vs Predicted Onion Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('hybrid_predictions.png')
    
    # Plot a zoomed-in view of the last 90 days
    if min_length > 90:
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates[-90:], y_true[-90:], label='Actual')
        plt.plot(test_dates[-90:], hybrid_preds[-90:], label='Hybrid Prediction')
        plt.title('Hybrid Model: Last 90 Days - Actual vs Predicted Onion Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig('hybrid_predictions_last_90_days.png')
    
    return rmse, mae, mape

def predict_future_price(date_str, prophet_model, lstm_model, scaler, features, df, seq_length=30, weights=(0.3, 0.7)):
    """Predict the onion price for a future date"""
    print(f"Predicting onion price for {date_str}...")
    
    # Convert string to datetime
    future_date = pd.to_datetime(date_str)
    
    # Prophet prediction
    # Create a future dataframe
    future_df = pd.DataFrame({'ds': [future_date]})
    
    # Add regressors (use average values for that month from historical data)
    month = future_date.month
    avg_rainfall = df[df['month'] == month]['rainfall'].mean()
    avg_monthly_price = df[df['month'] == month]['monthly_avg_price'].mean()
    
    future_df['rainfall'] = avg_rainfall
    future_df['monthly_avg_price'] = avg_monthly_price
    
    # Make prediction
    prophet_forecast = prophet_model.predict(future_df)
    prophet_pred = prophet_forecast['yhat'].values[0]
    
    # LSTM prediction
    # We need to create a complete feature set for the future date
    # First, get the last sequence from the data
    last_sequence = df[features].iloc[-seq_length:].copy()
    
    # Scale the sequence
    scaled_sequence = scaler.transform(last_sequence)
    
    # Reshape for LSTM [1, time steps, features]
    X = scaled_sequence.reshape(1, seq_length, len(features))
    
    # Make prediction
    lstm_pred = lstm_model.predict(X)[0][0]
    
    # Hybrid prediction
    hybrid_pred = weights[0] * prophet_pred + weights[1] * lstm_pred
    
    print(f"Prophet prediction: {prophet_pred:.2f}")
    print(f"LSTM prediction: {lstm_pred:.2f}")
    print(f"Hybrid prediction: {hybrid_pred:.2f}")
    
    return hybrid_pred

def save_models(prophet_model, lstm_model, features):
    """Save the trained models"""
    print("Saving models...")
    
    # Save Prophet model using pickle
    with open('prophet_model.pkl', 'wb') as f:
        pickle.dump(prophet_model, f)
    
    # Save LSTM model
    lstm_model.save('lstm_model.h5')
    
    # Save features list
    with open('lstm_features.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    print("Models saved successfully.")

def find_optimal_weights(prophet_preds, lstm_preds, y_true):
    """Find the optimal weights for the hybrid model"""
    print("Finding optimal weights for hybrid model...")
    
    best_rmse = float('inf')
    best_weights = (0.5, 0.5)
    
    # Try different weight combinations
    for w1 in np.arange(0, 1.1, 0.1):
        w2 = 1 - w1
        weights = (w1, w2)
        
        # Generate hybrid predictions
        hybrid_preds = w1 * prophet_preds + w2 * lstm_preds
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_true, hybrid_preds))
        
        print(f"Weights: {weights}, RMSE: {rmse:.2f}")
        
        # Update best weights if current is better
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = weights
    
    print(f"Optimal weights: {best_weights}, RMSE: {best_rmse:.2f}")
    return best_weights

def main():
    """Main function to train and evaluate models"""
    # Load data
    df = load_data()
    
    # Split data
    train_data, test_data = train_test_split(df)
    
    # Train Prophet model
    prophet_model = train_prophet_model(train_data)
    
    # Train LSTM model
    lstm_model, scaler, features, history = train_lstm_model(train_data)
    
    # Evaluate Prophet model
    prophet_forecast, prophet_rmse, prophet_mae, prophet_mape = evaluate_prophet_model(prophet_model, test_data)
    
    # Evaluate LSTM model
    lstm_preds, lstm_rmse, lstm_mae, lstm_mape = evaluate_lstm_model(lstm_model, scaler, features, test_data)
    
    # Generate hybrid predictions
    seq_length = 30
    # Make sure we're using the correct slice of prophet predictions
    prophet_preds = prophet_forecast['yhat'].values[seq_length:seq_length+len(lstm_preds)]
    
    # Ensure the lengths match
    min_length = min(len(prophet_preds), len(lstm_preds))
    prophet_preds = prophet_preds[:min_length]
    lstm_preds = lstm_preds[:min_length]
    y_true = test_data['value'].iloc[seq_length:seq_length+min_length].values
    
    # Find optimal weights
    optimal_weights = find_optimal_weights(prophet_preds, lstm_preds, y_true)
    
    # Generate hybrid predictions with optimal weights
    hybrid_preds = hybrid_predictions(prophet_preds, lstm_preds, weights=optimal_weights)
    
    # Evaluate hybrid model
    hybrid_rmse, hybrid_mae, hybrid_mape = evaluate_hybrid_model(hybrid_preds, test_data, seq_length, len(lstm_preds))
    
    # Save models
    save_models(prophet_model, lstm_model, features)
    
    # Example: Predict price for a future date
    future_date = '2025-05-01'
    predicted_price = predict_future_price(future_date, prophet_model, lstm_model, scaler, features, df, weights=optimal_weights)
    print(f"Predicted onion price for {future_date}: ₹{predicted_price:.2f}")
    
    # Create a summary of model performance
    models = ['Prophet', 'LSTM', 'Hybrid']
    rmse_values = [prophet_rmse, lstm_rmse, hybrid_rmse]
    mae_values = [prophet_mae, lstm_mae, hybrid_mae]
    mape_values = [prophet_mape, lstm_mape, hybrid_mape]
    
    summary = pd.DataFrame({
        'Model': models,
        'RMSE': rmse_values,
        'MAE': mae_values,
        'MAPE (%)': mape_values
    })
    
    print("\nModel Performance Summary:")
    print(summary)
    summary.to_csv('model_performance.csv', index=False)

if __name__ == "__main__":
    main()
