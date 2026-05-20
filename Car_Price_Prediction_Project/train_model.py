import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# load dataset
df = pd.read_csv("carData.csv")

# features (NOW INCLUDING MORE COLUMNS)
X = df[['year', 'kms_driven', 'company', 'name', 'fuel_type']]

y = df['Price']

# convert categorical to numeric (VERY IMPORTANT)
X = pd.get_dummies(X, drop_first=True)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = LinearRegression()
model.fit(X_train, y_train)

# save columns also (IMPORTANT for prediction)
pickle.dump(model, open("LinearRegressionModel.pkl", "wb"))
pickle.dump(X.columns, open("model_columns.pkl", "wb"))

print("✅ Model trained with company + model + fuel")