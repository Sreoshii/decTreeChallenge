import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
sales_data = pd.read_csv("salesPriceData.csv")

# Prepare model data (treating zipCode as numerical)
model_vars = ['SalePrice', 'LotArea', 'YearBuilt', 'GrLivArea', 'FullBath', 
              'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'zipCode']
model_data = sales_data[model_vars].dropna()

# Split data
X = model_data.drop('SalePrice', axis=1)
y = model_data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Build decision tree
print("Building decision tree with zipCode as numerical...")
tree_model = DecisionTreeRegressor(max_depth=3, 
                                  min_samples_split=20, 
                                  min_samples_leaf=10, 
                                  random_state=123)
tree_model.fit(X_train, y_train)

print(f"Model built with {tree_model.get_n_leaves()} terminal nodes")

# Visualize tree
print("Visualizing tree...")
plt.figure(figsize=(10, 6))
plot_tree(tree_model, 
          feature_names=X_train.columns,
          filled=True, 
          rounded=True,
          fontsize=10,
          max_depth=3)
plt.title("Decision Tree (zipCode as Numerical)")
plt.tight_layout()
plt.savefig("index_files/figure-html/visualize-tree-python-output-1.png", dpi=150)
plt.close()

# Extract and display feature importance
print("Calculating feature importance...")
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': tree_model.feature_importances_
}).sort_values('Importance', ascending=False)

importance_df['Importance_Percent'] = (importance_df['Importance'] * 100).round(2)

# Check zipCode ranking
zipcode_rank = importance_df[importance_df['Feature'] == 'zipCode'].index[0] + 1
zipcode_importance = importance_df[importance_df['Feature'] == 'zipCode']['Importance_Percent'].iloc[0]

print("\nFeature Importance (zipCode as Numerical):")
print(importance_df)
print(f"\nzipCode rank: {zipcode_rank}")
print(f"zipCode importance: {zipcode_importance}%")

# Plot feature importance
plt.figure(figsize=(8, 5))
plt.barh(range(len(importance_df)), importance_df['Importance'], 
         color='steelblue', alpha=0.7)
plt.yticks(range(len(importance_df)), importance_df['Feature'])
plt.xlabel('Importance Score')
plt.title('Feature Importance (zipCode as Numerical)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("index_files/figure-html/importance-plot-python-output-1.png", dpi=150)
plt.close()

# Now do categorical encoding
print("\n" + "="*50)
print("Now building model with zipCode as categorical...")

# One-hot encode zipCode
zipcode_encoded = pd.get_dummies(model_data['zipCode'], prefix='zipCode')
model_data_cat = pd.concat([model_data.drop('zipCode', axis=1), zipcode_encoded], axis=1)

# Split data
X_cat = model_data_cat.drop('SalePrice', axis=1)
y_cat = model_data_cat['SalePrice']
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y_cat, test_size=0.2, random_state=123)

# Build decision tree with one-hot encoded zipCode
tree_model_cat = DecisionTreeRegressor(max_depth=3, 
                                      min_samples_split=20, 
                                      min_samples_leaf=10, 
                                      random_state=123)
tree_model_cat.fit(X_train_cat, y_train_cat)

print(f"Categorical model built with {tree_model_cat.get_n_leaves()} terminal nodes")

# Feature importance with one-hot encoded zipCode
importance_cat_df = pd.DataFrame({
    'Feature': X_train_cat.columns,
    'Importance': tree_model_cat.feature_importances_
}).sort_values('Importance', ascending=False)

importance_cat_df['Importance_Percent'] = (importance_cat_df['Importance'] * 100).round(2)

# Check zipCode features
zipcode_features = [col for col in X_train_cat.columns if col.startswith('zipCode')]
zipcode_importance_total = importance_cat_df[importance_cat_df['Feature'].isin(zipcode_features)]['Importance'].sum()
total_importance = importance_cat_df['Importance'].sum()
zipcode_percent = (zipcode_importance_total / total_importance * 100).round(2)

print("\nFeature Importance (zipCode as Categorical):")
print(importance_cat_df.head(20))
print(f"\nTotal zipCode importance: {zipcode_percent}%")
print(f"Number of zipCode features: {len(zipcode_features)}")

# Visualize tree with one-hot encoded zipCode
plt.figure(figsize=(12, 8))
plot_tree(tree_model_cat, 
          feature_names=X_train_cat.columns,
          filled=True, 
          rounded=True,
          fontsize=8,
          max_depth=4)
plt.title("Decision Tree (zipCode One-Hot Encoded)")
plt.tight_layout()
plt.savefig("index_files/figure-html/visualize-tree-cat-python-output-1.png", dpi=150)
plt.close()

# Plot feature importance for categorical zipCode
plt.figure(figsize=(8, 5))
plt.barh(range(len(importance_cat_df)), importance_cat_df['Importance'], 
         color='darkgreen', alpha=0.7)
plt.yticks(range(len(importance_cat_df)), importance_cat_df['Feature'])
plt.xlabel('Importance Score')
plt.title('Feature Importance (zipCode One-Hot Encoded)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("index_files/figure-html/importance-plot-cat-python-output-1.png", dpi=150)
plt.close()

print("\nAnalysis complete! All plots saved.")

