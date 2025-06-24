import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# ✅ Step 1: Load training data
df = pd.read_csv("train.csv")
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'SalePrice']
df = df[features]

# ✅ Step 2: Split into inputs and target
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 3: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ Step 4: Predict on validation set
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)

# ✅ Step 5: Plot actual vs predicted
plt.scatter(y_val, y_pred, alpha=0.5, color='teal')
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# ✅ Step 6: Predict on test data and save to CSV
test_df = pd.read_csv("test.csv")
test_features = test_df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']].fillna(0)
test_preds = model.predict(test_features)

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_preds
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv has been saved!")
