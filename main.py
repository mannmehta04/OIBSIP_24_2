import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('./data/unemployment.csv')
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
df['Region'] = pd.Categorical(df['Region']).codes
df['Area'] = pd.Categorical(df['Area']).codes
df.fillna(df.mean(), inplace=True)

X = df[['Region', 'Estimated Employed', 'Estimated Labour Participation Rate (%)', 'Area']]
y = df['Estimated Unemployment Rate (%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Regressor Plot
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Actual vs Predicted Unemployment Rate')
plt.xlabel('Index')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.grid(True)
mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(f"Index: {sel.target[0]}\nUnemployment Rate: {sel.target[1]:.2f}%"))
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Actual Unemployment Rate (%)')
plt.ylabel('Residuals')
plt.grid(True)
mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(f"Actual Unemployment Rate: {sel.target[0]:.2f}%\nResidual: {sel.target[1]:.2f}%"))
plt.show()

# Feature Importance Plot
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X.columns, feature_importances)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.grid(True)
plt.show()

# Distribution Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test, label='Actual', shade=True)
sns.kdeplot(y_pred, label='Predicted', shade=True)
plt.title('Distribution of Actual vs Predicted Unemployment Rate')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Unemployment Rate')
plt.xlabel('Actual Unemployment Rate (%)')
plt.ylabel('Predicted Unemployment Rate (%)')
plt.grid(True)
mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(f"Actual: {sel.target[0]:.2f}%\nPredicted: {sel.target[1]:.2f}%"))
plt.show()
