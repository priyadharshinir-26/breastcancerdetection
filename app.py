import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve

# Title of the Streamlit App
st.title("Breast Cancer Classification with AdaBoost")

# Dataset URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
data = pd.read_csv(url, header=None, names=column_names)
data.drop(columns=['id'], inplace=True)
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})

# Splitting data
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# AdaBoost without hyperparameter tuning
adaboost = AdaBoostClassifier(random_state=42)
adaboost.fit(X_train_sm, y_train_sm)
y_pred = adaboost.predict(X_test)

# Classification Report (Before Hyperparameter Tuning)
report_before_tuning = classification_report(y_test, y_pred, output_dict=True)

# AdaBoost with hyperparameter tuning
param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}
grid_search = GridSearchCV(estimator=AdaBoostClassifier(random_state=42),
                           param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_sm, y_train_sm)

# Best Model
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

# Classification Report (After Hyperparameter Tuning)
report_after_tuning = classification_report(y_test, y_pred_tuned, output_dict=True)

# Feature Importances
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
most_important_feature = feature_importance_df.iloc[0]

# Streamlit Sections
st.header("Classification Report")
st.subheader("Before Hyperparameter Tuning")
st.text(classification_report(y_test, y_pred))

st.subheader("After Hyperparameter Tuning")
st.text(classification_report(y_test, y_pred_tuned))

st.header("Feature Importances")
st.write(feature_importance_df)

# Highlight the most important feature
st.subheader("Most Important Feature")
st.write(f"The most important feature is **{most_important_feature['Feature']}** "
         f"with an importance of **{most_important_feature['Importance']*100:.2f}%**.")

# Display Feature Importance Plot
st.subheader("Feature Importance Plot")
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Fine-Tuned AdaBoost Model')
st.pyplot(plt)
