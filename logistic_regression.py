import sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

X = np.load('C:/Melo/Project37/bladdercancer/data_features/X_train_b7.npy')
y = np.load('C:/Melo/Project37/bladdercancer/data_features/y_train_b7.npy')

# Split data and train classifier
print(f"Training data shape: {X.shape}, labels shape: {y.shape}")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=1337)
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)


# Check on validation
val_preds= clf.predict_proba(X_val)[:,1]
print(f"On validation set:")
print(f"Accuracy: {clf.score(X_val, y_val)}")
print(f"LOG LOSS: {log_loss(y_val, val_preds)} ")
print("%--------------------------------------------------%")

# Get predictions on test set
print("Getting predictions for test set")
X_test = np.load('C:/Melo/Project37/bladdercancer/data_features/X_test_b7.npy')
X_test_preds = clf.predict_proba(X_test)[:,1]
df = pd.DataFrame({'id': np.arange(1, 81), 'label': np.clip(X_test_preds, 0.005, 0.995)})
df.to_csv('C:/Melo/Project37/bladdercancer/submission.csv', index=False)
print("Done getting predictions!")
