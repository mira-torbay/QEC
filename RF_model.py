import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# read .csv
data = pd.read_csv("example_random_with_trends.csv", parse_dates=['timestamp'])

# set timestamp as the index
data.set_index('timestamp', inplace=True)

print(data.head())
print(data.info())

# resample to hourly data, create time features
data_hourly = data.resample('H').mean()
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek

# clean data (fill missing values, resampling intervals)
data = data.resample('30T').asfreq()  # resample to 30 minutes
data['sleep_duration'] = data['sleep_duration'].fillna(method='ffill')
data['caffeine'] = data['caffeine'].fillna(method='ffill')
data['stress_level'] = data['stress_level'].fillna(method='ffill')
data = data.dropna(subset=['sleep_duration', 'caffeine', 'stress_level'])

# add lag features (previous values) to deal with patterns over time
data['prev_stress_level'] = data['stress_level'].shift(1)
data['prev_caffeine'] = data['caffeine'].shift(1)
data['prev_HR'] = data['heart_rate'].shift(1)
data = data.dropna(subset=['prev_stress_level', 'prev_caffeine', 'prev_HR'])

# scale continuous features
scaler = StandardScaler()
data[['heart_rate', 'sleep_duration', 'skin_temp', 'caffeine', 'blood_oxygen']] = scaler.fit_transform(
    data[['heart_rate', 'sleep_duration', 'skin_temp', 'caffeine', 'blood_oxygen']])

# define features (X) and target (y)
X = data[['heart_rate', 'sleep_duration', 'skin_temp', 'caffeine', 'blood_oxygen',
          'prev_stress_level', 'prev_caffeine', 'prev_HR']]
y = data['stress_level']

# split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# initialize Random Forest model
model = RandomForestRegressor(n_estimators=100, max_features='sqrt')

# hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 4, 8],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# best parameters from GridSearchCV
print(f"Best Parameters: {grid_search.best_params_}")

# get best model
best_model = grid_search.best_estimator_

# make predictions, evaluate accuracy
predictions = best_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae}")

# cross-validation to evaluate model
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-validated Mean Absolute Error: {-cv_scores.mean()}")

# importances of each factor affected stress
importances = best_model.feature_importances_

# create a DataFrame with each factor and its corresponding importance
factor_importances = pd.DataFrame({
    'Factor': X.columns,
    'Importance': importances
})

# sort the DataFrame by importance in descending order
factor_importances = factor_importances.sort_values(by='Importance', ascending=False)

# print the ranked factors
print("Ranking of factors by effect on stress level:")
print(factor_importances)

# combine predictions with the timestamp to output clear results
result = X_test.copy()
result['timestamp'] = X_test.index
result['predicted_stress_level'] = predictions

print(result[['timestamp', 'heart_rate', 'sleep_duration', 'skin_temp', 'caffeine', 'blood_oxygen', 'predicted_stress_level']])

# plot Actual vs Predicted Stress Levels
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.7, edgecolors="w", linewidth=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Diagonal line
plt.title('Actual vs Predicted Stress Levels')
plt.xlabel('Actual Stress Level')
plt.ylabel('Predicted Stress Level')
plt.grid(True)
plt.show()
