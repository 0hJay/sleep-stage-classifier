import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Placeholder for data loading (you'll replace this with real Apple Watch data later)
def load_data():
    # Simulated data: accel, heart rate, HRV, respiratory rate, circadian time
    data = {
        'accel_magnitude': np.random.normal(0.05, 0.02, 1000),  # Simulated movement
        'heart_rate': np.random.normal(60, 5, 1000),  # Simulated HR
        'hrv': np.random.normal(30, 5, 1000),  # Simulated HRV (ms)
        'respiratory_rate': np.random.normal(15, 2, 1000),  # From watchOS 11
        'circadian_time': np.random.uniform(0, 24*3600, 1000),  # Seconds since last sleep
        'stage': np.random.choice(['wake', 'NREM', 'REM'], 1000, p=[0.2, 0.6, 0.2])
    }
    return pd.DataFrame(data)

# Feature extraction
def extract_features(df):
    features = df[['accel_magnitude', 'heart_rate', 'hrv', 'respiratory_rate', 'circadian_time']]
    labels = df['stage']
    return features, labels

# Train the model
def train_model():
    df = load_data()
    X, y = extract_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"F1-score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
    
    return model

if __name__ == "__main__":
    model = train_model()
