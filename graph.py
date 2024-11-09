import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read .csv
# load .csv and parse the timestamp as a datetime
data = pd.read_csv("example.csv", parse_dates=['timestamp'])

# set the timestamp as the index
data.set_index('timestamp', inplace=True)

# Plotting settings
sns.set(style="whitegrid")  # Set the style of the plots

# List of factors to plot against stress level
factors = ['heart_rate', 'sleep_duration', 'skin_temp', 'caffeine', 'blood_oxygen']

# Set up the plot grid (2 rows and 3 columns for a clean layout)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Flatten the axes for easy iteration
axes = axes.flatten()

# Loop over the factors and plot each
for i, factor in enumerate(factors):
    sns.scatterplot(x=data[factor], y=data['stress_level'], ax=axes[i], color='b')
    axes[i].set_title(f'{factor} vs Stress Level')
    axes[i].set_xlabel(f'{factor}')
    axes[i].set_ylabel('Stress Level')

# Adjust layout and show plot
plt.tight_layout()
plt.show()
