# app/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load the feature dataset
df = pd.read_csv("app/feature_dataset_v2.csv")
print(f"Loaded dataset with {len(df)} rows")
print(df.head())  # Optional: just to verify visually


# 2. Ensure label is integer (if already 0 and 1)
df['label'] = df['label'].astype(int)
print("✅ Label value counts:")
print(df['label'].value_counts())



# 3. Define features and target
X = df.drop("label", axis=1)
y = df["label"]


print("🧪 Sanity Check – Shape of Features (X) and Labels (y):")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Any NaNs in X? {X.isnull().any().any()}")
print(f"Any NaNs in y? {y.isnull().any()}")

print("📊 Column Data Types in X:")
print(X.dtypes)

print("🧼 Are any rows fully numeric?")
print(X.applymap(lambda x: isinstance(x, (int, float))).all(axis=1).value_counts())

# Optional: save and check the dataframe
X.to_csv("debug_X.csv", index=False)

# 4. Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")



# 5. Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Predict on test set
y_pred = model.predict(X_test)

# 7. Evaluate performance
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Save the trained model
joblib.dump(model, "app/verdict_model.pkl")
print("🧠 Model saved to app/verdict_model.pkl")
