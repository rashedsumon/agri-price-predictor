import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def preprocess_and_train(df):
    """
    Transforms raw long-format time series price data, extracts features,
    and trains an AI regression model on the historical averages.
    """
    # Clean column values & treat dates
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Strip whitespace out of strings
    df['Commodity'] = df['Commodity'].str.strip()
    
    # Focus purely on standard KG sales to normalize distribution scale metrics
    df['Unit'] = df['Unit'].str.upper().str.strip()
    df = df[df['Unit'] == 'KG']
    
    # Feature Engineering via Date component breakdowns
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Convert 'Commodity' categories into numerical structures using Label/Factorize Mapping
    df['Commodity_Code'], unique_commodities = pd.factorize(df['Commodity'])
    commodity_mapping = dict(zip(unique_commodities, range(len(unique_commodities))))
    
    # Define features and target variable
    X = df[['Commodity_Code', 'Year', 'Month', 'Day', 'DayOfWeek']]
    y = df['Average']
    
    # Train test split for quick validation sanity test evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and fit model
    print("Training RandomForest Regression Predictive Engine...")
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    return model, commodity_mapping, df

def predict_future_price(model, commodity_mapping, commodity_name, target_date):
    """
    Constructs an input array dynamically from user selection vectors to predict pricing indexes.
    """
    if commodity_name not in commodity_mapping:
        return None
        
    comp_date = pd.to_datetime(target_date)
    comm_code = commodity_mapping[commodity_name]
    
    # Map layout precisely matches structural fit pattern dimensions:
    # ['Commodity_Code', 'Year', 'Month', 'Day', 'DayOfWeek']
    input_data = np.array([[
        comm_code,
        comp_date.year,
        comp_date.month,
        comp_date.day,
        comp_date.dayofweek
    ]])
    
    prediction = model.predict(input_data)
    return prediction[0]