import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

healthy_data = pd.read_csv('C:\CBIO\CBIO_Hackathon\GeneMatrix-Healthy.csv',index_col=0).T
sick_data = pd.read_csv('C:\CBIO\CBIO_Hackathon\GeneMatrix-LLB.csv',index_col=0).T


healthy_data['label'] = 0
sick_data['label']= 1


all_data = pd.concat([healthy_data, sick_data], axis=0)


print("Shape of all_data:", all_data.shape)


X = all_data.drop('label', axis=1)
y = all_data['label']


print("Healthy samples:", sum(y == 0))
print("Sick samples:   ", sum(y == 1))


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators':      [50, 100, 200],
    'max_depth':         [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print("\nBest parameters found by GridSearch:")
print(grid_search.best_params_)


print("\nBest score (CV):", grid_search.best_score_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_scaled)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

feature_importances = best_model.feature_importances_
features = X.columns

fi_df = pd.DataFrame({
    'feature': features,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print("\n=== Top 10 most important features ===")
print(fi_df.head(10))
