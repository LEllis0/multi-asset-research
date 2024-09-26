import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier, plot_tree
from backtesting import Backtest, Strategy
from datetime import datetime

# Initial data fetching
df = yf.download("AAPL")
df = df.loc['2016-12-08':,:]

# Calculate percentage change for tomorrow
df['change_tomorrow'] = df['Close'].pct_change(-1) * 100 * -1
df = df.dropna().copy()  # Drop NaN values

# Create binary classification target: 'Up' or 'Down'
df['change_tomorrow_direction'] = np.where(df['change_tomorrow'] > 0, 'Up', 'Down')

# Prepare the explanatory data: Drop columns that are not features
explanatory = df.drop(columns=['change_tomorrow', 'change_tomorrow_direction'])

# Decision tree model
model_dt = DecisionTreeClassifier(max_depth=15)  # Limit tree depth to avoid overfitting
model_dt.fit(X=explanatory, y=df['change_tomorrow_direction'])

# Plot the decision tree (optional)
plot_tree(decision_tree=model_dt, feature_names=explanatory.columns)

# Predict the entire dataset
y_pred = model_dt.predict(X=explanatory)

# Create a DataFrame to compare actual vs predicted
df_predictions = df[['change_tomorrow_direction']].copy()
df_predictions['predictions'] = y_pred

# Check model accuracy
model_accuracy = model_dt.score(X=explanatory, y=df['change_tomorrow_direction'])
print(f"Model Accuracy: {model_accuracy:.2f}")

# Today's explanatory data
explanatory_today = explanatory.iloc[[-1], :]
prediction = model_dt.predict(explanatory_today)[0]
print(f"Prediction for tomorrow: {prediction}")

# Define custom strategy for backtesting
class ClassificationUp(Strategy):  
    def init(self):
        self.model = model_dt  # Use the decision tree model

    def next(self):
        # Drop target columns for prediction
        explanatory_today = self.data.df.iloc[[-1], :].drop(columns=['change_tomorrow', 'change_tomorrow_direction'])
        
        forecast_tomorrow = self.model.predict(explanatory_today)[0]  # Predict tomorrow's direction
        
        if forecast_tomorrow == 'Up':  
            self.buy()  # Buy if prediction is 'Up'
        elif forecast_tomorrow == 'Down':  
            self.sell()  # Sell if prediction is 'Down'

# Backtesting
bt = Backtest(df, ClassificationUp, cash=10000)  # Set starting cash
stats = bt.run()
print(stats)