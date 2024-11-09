import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# read .csv
# load .csv and parse the timestamp as a datetime
data = pd.read_csv("example_random_with_trends.csv", parse_dates=['timestamp'])

# set the timestamp as the index
data.set_index('timestamp', inplace=True)

print(data.head())  # displays first 5 rows
print(data.info())

data_hourly = data.resample('H').mean()  # take the average of each data category each hour
data['hour'] = data.index.hour  # extract hour from timestamp
data['day_of_week'] = data.index.dayofweek  # extract day of the week

# clean data
data = data.resample('30T').asfreq()  # resample to 30 minutes
data['sleep_duration'] = data['sleep_duration'].fillna(method='ffill')  # for sleep duration, caffeine and stress level, replace missing value with most recent
data['caffeine'] = data['caffeine'].fillna(method='ffill')
data['stress_level'] = data['stress_level'].fillna(method='ffill')
data = data.dropna(subset=['sleep_duration'])  # drop rows if NaNs still present (no value to use for ffill)
data = data.dropna(subset=['caffeine'])
data = data.dropna(subset=['stress_level'])
data = data.apply(lambda col: col.fillna((col.shift(1) + col.shift(-1)) / 2) if col.name != 'sleep_duration' else col, axis=0)  # for everything else, take mean of next and prev values


# scale continuous features
scaler = StandardScaler()
data[['heart_rate', 'sleep_duration', 'skin_temp', 'caffeine', 'blood_oxygen']] = scaler.fit_transform(
    data[['heart_rate', 'sleep_duration', 'skin_temp', 'caffeine', 'blood_oxygen']])

# define features (X) and target (y)
X = data[['heart_rate', 'sleep_duration', 'skin_temp', 'caffeine', 'blood_oxygen']]
y = data['stress_level']

# split data into training and testing
train_size = int(len(data) * 0.7)
train_data = data[:train_size]
test_data = data[train_size:]

X_train = train_data[['heart_rate', 'sleep_duration', 'skin_temp', 'caffeine', 'blood_oxygen']]
y_train = train_data['stress_level']
X_test = test_data[['heart_rate', 'sleep_duration', 'skin_temp', 'caffeine', 'blood_oxygen']]
y_test = test_data['stress_level']

# initialize and train a Random Forest model
model = RandomForestRegressor(n_estimators=100)  # initialize with 100 decision trees
model.fit(X_train, y_train)  # train
predictions = model.predict(X_test)  # make predictions on test data

# combine predictions with the timestamp
result = X_test.copy()
result['timestamp'] = X_test.index
result['predicted_stress_level'] = predictions

print(result[['timestamp', 'heart_rate', 'sleep_duration', 'skin_temp', 'caffeine', 'blood_oxygen', 'predicted_stress_level']])

# get feature importances
importances = model.feature_importances_

# create a DataFrame with each factor and it's corresponding importance (relative effect on stress)
factor_importances = pd.DataFrame({
    'Factor': X.columns,
    'Importance': importances
})

# Sort the DataFrame by importance in descending order
factor_importances = factor_importances.sort_values(by='Importance', ascending=False)

# Print the ranked factors
print("Ranking of factors by effect on stress level:")
print(factor_importances)
