import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load your dataset
# Update the path to your actual file location
df = pd.read_csv('dataset/HistoricalQuotes.csv')

# Clean column names if needed
df.columns = df.columns.str.strip()

# Convert Date to datetime and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# Clean and prepare price data
# Handle different possible column names for Close price
close_col = None
for col in ['Close/Last', 'Close', 'close']:
    if col in df.columns:
        close_col = col
        break

if close_col is None:
    print("Error: Could not find Close price column")
    exit()

# Remove $ signs and convert to numeric if needed
if df[close_col].dtype == 'object':
    df[close_col] = df[close_col].str.replace('$', '').astype(float)

# Do the same for other price columns
price_cols = ['Open', 'High', 'Low', 'Volume']
for col in price_cols:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].str.replace('$', '').str.replace(',', '').astype(float)

# Create Close column for consistency
df['Close'] = df[close_col]

# Calculate next day's price change percentage
df['Next_Day_Close'] = df['Close'].shift(-1)
df['Price_Change_Pct'] = (df['Next_Day_Close'] - df['Close']) / df['Close']

# Multi-class classification with threshold
threshold = 0.01  # 1% threshold

df['Price_Direction'] = np.where(df['Price_Change_Pct'] > threshold, 2,      # Up
                                np.where(df['Price_Change_Pct'] < -threshold, 0, 1))  # Down, Sideways

# Create labels for better interpretation
direction_labels = {0: 'Down', 1: 'Sideways', 2: 'Up'}
df['Direction_Label'] = df['Price_Direction'].map(direction_labels)

print("\nMulti-Class Price Direction Distribution:")
print(df['Price_Direction'].value_counts().sort_index())
print("\nPercentage Distribution:")
print(df['Price_Direction'].value_counts(normalize=True).sort_index())

# Technical indicators
df['Price_Change'] = df['Close'].pct_change()
df['Volume_Change'] = df['Volume'].pct_change()

# Moving averages
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['Price_Above_SMA5'] = (df['Close'] > df['SMA_5']).astype(int)
df['Price_Above_SMA10'] = (df['Close'] > df['SMA_10']).astype(int)

# Volatility indicators
df['Volatility'] = df['Price_Change'].rolling(window=5).std()
df['High_Low_Ratio'] = df['High'] / df['Low']
df['Open_Close_Ratio'] = df['Open'] / df['Close']

# Price position within the day's range
df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])

# Previous day's performance
df['Prev_Day_Up'] = (df['Close'] > df['Close'].shift(1)).astype(int)
df['Prev_Day_Volume'] = df['Volume'].shift(1)

# Momentum indicators
df['Price_Momentum_3'] = df['Close'] / df['Close'].shift(3)
df['Volume_Momentum_3'] = df['Volume'] / df['Volume'].shift(3)

print("Features created successfully!")

# Select relevant features for direction prediction
features = [
    'Open', 'High', 'Low', 'Volume',
    'Price_Change', 'Volume_Change',
    'Price_Above_SMA5', 'Price_Above_SMA10',
    'Volatility', 'High_Low_Ratio', 'Open_Close_Ratio',
    'Price_Position', 'Prev_Day_Up',
    'Price_Momentum_3', 'Volume_Momentum_3'
]

# Remove rows with NaN values
df_clean = df.dropna()
print(f"Dataset shape after cleaning: {df_clean.shape}")

# Prepare features and target
X = df_clean[features]
y = df_clean['Price_Direction']

print(f"Final features shape: {X.shape}")
print(f"Multi-class target distribution after cleaning:")
print(y.value_counts(normalize=True).sort_index())

# Time-based split (important for time series)
split_index = int(len(df_clean) * 0.80)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

print(f"\nTraining set: {X_train.shape[0]} samples (80%)")
print(f"Test set: {X_test.shape[0]} samples (20%)")

# Create decision tree classifier for multi-class
model = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print("Multi-class model trained successfully!")

# Cross-validation with TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

for train_idx, test_idx in tscv.split(X):
    X_train_cv = X.iloc[train_idx]
    X_test_cv = X.iloc[test_idx]
    y_train_cv = y.iloc[train_idx]
    y_test_cv = y.iloc[test_idx]
    
    model.fit(X_train_cv, y_train_cv)
    y_pred_cv = model.predict(X_test_cv)
    cv_scores.append(accuracy_score(y_test_cv, y_pred_cv))

print(f"\nCross-validation accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy:.4f}")

# Detailed classification report
target_names = ['Down', 'Sideways', 'Up']
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("       Down  Sideways  Up")
for i, row in enumerate(cm):
    print(f"{target_names[i]:>6}: {row}")

# Class-specific metrics
print("\nClass-specific Performance:")
for i, class_name in enumerate(target_names):
    class_precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
    class_recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
    print(f"{class_name}: Precision={class_precision:.4f}, Recall={class_recall:.4f}")

# Trading strategy analysis
print("\nTrading Strategy Analysis:")

# UP predictions
up_predictions = (y_pred == 2)
if up_predictions.sum() > 0:
    up_accuracy = (y_test[up_predictions] == 2).mean()
    print(f"When predicting UP: Accuracy = {up_accuracy:.4f}")

# DOWN predictions
down_predictions = (y_pred == 0)
if down_predictions.sum() > 0:
    down_accuracy = (y_test[down_predictions] == 0).mean()
    print(f"When predicting DOWN: Accuracy = {down_accuracy:.4f}")

# High confidence predictions
max_proba = y_pred_proba.max(axis=1)
high_confidence = max_proba > 0.6

if high_confidence.sum() > 0:
    high_conf_accuracy = (y_test[high_confidence] == y_pred[high_confidence]).mean()
    print(f"High confidence predictions (>60%): Accuracy = {high_conf_accuracy:.4f}")
    print(f"High confidence predictions: {high_confidence.sum()}/{len(y_test)} = {high_confidence.mean():.2%}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Recent predictions
if len(X_test) >= 5:
    last_5_days = X_test.tail(5)
    predictions = model.predict(last_5_days)
    probabilities = model.predict_proba(last_5_days)
    
    results = pd.DataFrame({
        'Date': df_clean['Date'].iloc[-5:].values,
        'Actual_Close': df_clean['Close'].iloc[-5:].values,
        'Predicted_Direction': [target_names[p] for p in predictions],
        'Prob_Down': probabilities[:, 0],
        'Prob_Sideways': probabilities[:, 1],
        'Prob_Up': probabilities[:, 2],
        'Actual_Direction': [target_names[a] for a in y_test.tail(5)]
    })
    
    print("\nRecent Multi-Class Predictions:")
    print(results.round(3))

# Future predictions
last_date = df_clean['Date'].iloc[-1]
print(f"\nLast date in dataset: {last_date}")

future_dates = []
for i in range(1, 6):
    future_date = last_date + timedelta(days=i)
    future_dates.append(future_date)

# Use last available features for prediction
last_features = X_test.iloc[-1:].values
predictions = model.predict([last_features[0]] * 5)
probabilities = model.predict_proba([last_features[0]] * 5)

future_results = pd.DataFrame({
    'Date': future_dates,
    'Predicted_Direction': [target_names[p] for p in predictions],
    'Prob_Down': probabilities[:, 0],
    'Prob_Sideways': probabilities[:, 1],
    'Prob_Up': probabilities[:, 2]
})

print("\nFuture Predictions:")
print(future_results.round(3))

# Decision tree visualization (optional - may be too large)
try:
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=X_train.columns, 
              class_names=target_names, fontsize=8, max_depth=3)
    plt.title("Decision Tree Visualization (First 3 Levels)")
    plt.show()
except:
    print("\nDecision tree visualization skipped (tree too large or display issue)")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)