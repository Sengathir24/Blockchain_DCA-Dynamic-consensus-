import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\Downloads\blockchain_traffic_trafficsim.csv")

# Drop unnecessary columns
df = df.drop(columns=['row_id', 'timestamp'], errors='ignore')

# Encode target
label_encoder = LabelEncoder()
df['consensus_encoded'] = label_encoder.fit_transform(df['consensus'])

# Keep only numeric features
X = df.select_dtypes(include=[np.number]).drop(columns=['consensus_encoded'])
y = df['consensus_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model & label encoder
joblib.dump(clf, "decision_tree_consensus.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("💾 Model and encoder saved successfully!")
