# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv('D:\\ML PROJECTS\\AML lab\\dataset\\HistoricalQuotes.csv')

# %%
# Check exact column names
print("Column names:")
print(df.columns.tolist())

# Check for any extra spaces
print("\nColumn names with quotes to see spaces:")
for col in df.columns:
    print(f"'{col}'")

# %%
# Fix column names first (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# Now convert to numeric
df['Close/Last'] = pd.to_numeric(df['Close/Last'].str.replace('$', ''))
df['Open'] = pd.to_numeric(df['Open'].str.replace('$', ''))
df['High'] = pd.to_numeric(df['High'].str.replace('$', ''))
df['Low'] = pd.to_numeric(df['Low'].str.replace('$', ''))
df['Date'] = pd.to_datetime(df['Date'])


# %%
# Function to clean and convert price columns
def clean_price_column(column):
    if column.dtype == 'object':
        # If it's object type, try to clean it
        return pd.to_numeric(column.astype(str).str.replace('$', '').str.replace(',', ''))
    else:
        # If it's already numeric, return as is
        return column

# Apply to price columns
df['Close/Last'] = clean_price_column(df['Close/Last'])
df['Open'] = clean_price_column(df['Open'])
df['High'] = clean_price_column(df['High'])
df['Low'] = clean_price_column(df['Low'])
df['Date'] = pd.to_datetime(df['Date'])

print("After conversion:")
print(df.dtypes)
print(df.head())

# %%
# Create target variable (e.g., price direction - will stock go up next day?)
df['Price_Up'] = (df['Close/Last'].shift(-1) > df['Close/Last']).astype(int)

# Create additional features
df['Price_Range'] = df['High'] - df['Low']
df['Price_Change'] = df['Close/Last'] - df['Open']
df['Price_Change_Pct'] = ((df['Close/Last'] - df['Open']) / df['Open']) * 100
df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
df['Close_MA_5'] = df['Close/Last'].rolling(window=5).mean()
df['Volume_Above_Avg'] = (df['Volume'] > df['Volume_MA_5']).astype(int)

# Check the new features
print("New features created:")
print(df[['Date', 'Close/Last', 'Price_Up', 'Price_Range', 'Price_Change', 'Volume_MA_5']].head(10))

# %%
# Select features for the model
feature_columns = ['Volume', 'Price_Range', 'Price_Change', 'Price_Change_Pct', 'Volume_MA_5', 'Volume_Above_Avg']

# Remove rows with NaN values (due to rolling averages and shifts)
df_clean = df.dropna()

X = df_clean[feature_columns]
y = df_clean['Price_Up']

print(f"Dataset shape: {X.shape}")
print(f"Target distribution:")
print(y.value_counts())

# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# %%
# Get feature importance (coefficients)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'coefficient': model.coef_[0],
    'abs_coefficient': abs(model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print("Feature Importance:")
print(feature_importance)

# Show some predictions with probabilities
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Probability': y_pred_proba
}).head(10)

print("\nSample Predictions:")
print(results_df)

# %%
# Create confusion matrix plot
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Price Down (0)', 'Price Up (1)'],
                yticklabels=['Price Down (0)', 'Price Up (1)'],
                cbar_kws={'label': 'Count'})
    
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Print additional metrics
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix Breakdown:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    
    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")

# Call the function
plot_confusion_matrix(y_test, y_pred, 'Stock Price Direction Prediction - Confusion Matrix')

# %%
# Feature Collection heatmap
def plot_feature_correlation(X, feature_names):
    plt.figure(figsize=(10, 8))
    correlation_matrix = np.corrcoef(X.T)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

plot_feature_correlation(X_train_scaled, feature_columns)


