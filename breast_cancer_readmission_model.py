
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Simulate dataset
np.random.seed(42)
n = 1024
data = pd.DataFrame({
    'age': np.random.randint(40, 85, size=n),
    'comorbidities': np.random.poisson(1.5, size=n),
    'ecog_score': np.random.choice([0, 1, 2, 3], size=n, p=[0.3, 0.4, 0.2, 0.1]),
    'dose_reduction_cycle1': np.random.binomial(1, 0.3, size=n),
    'baseline_neutrophil': np.round(np.random.normal(3.5, 1.2, size=n), 1),
})

logits = (
    0.02 * data['age'] +
    0.5 * data['comorbidities'] +
    0.8 * data['ecog_score'] +
    0.6 * data['dose_reduction_cycle1'] -
    0.4 * data['baseline_neutrophil']
)
prob = 1 / (1 + np.exp(-logits))
data['readmitted_30d'] = np.random.binomial(1, prob)

# Train model
X = data.drop('readmitted_30d', axis=1)
y = data['readmitted_30d']
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
y_prob = clf.predict_proba(X)[:, 1]

# Evaluate performance
roc_auc = roc_auc_score(y, y_prob)
print(f"ROC AUC: {roc_auc:.2f}")
