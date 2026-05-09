import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# Load dataset
data = pd.read_csv("dataset/students.csv")

# Encode categorical features
encoder = LabelEncoder()

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = encoder.fit_transform(data[col])


# Define features and target
X = data.drop("math score", axis=1)
y = data["math score"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

results = {}


# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    results[name] = r2

    print(f"\n{name}")
    print("R2 Score:", r2)
    print("Mean Absolute Error:", mae)


# Plot model comparison
plt.bar(results.keys(), results.values())
plt.title("Model Performance Comparison")
plt.ylabel("R2 Score")
plt.xlabel("Models")
plt.show()