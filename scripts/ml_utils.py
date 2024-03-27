from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def train_KNN_model(stock_data, features, target, model=KNeighborsClassifier()):
    X_train, y_train = stock_data[features], stock_data[target]
    model.fit(X_train, y_train)
    return model

def train_random_forest_model(stock_data, features, target, n_estimators=100, max_depth=None, random_state=42):
    X_train, y_train = stock_data[features], stock_data[target]
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_random_forest_regressor(stock_data, features, target, n_estimators=100, max_depth=None, random_state=42):
    X_train, y_train = stock_data[features], stock_data[target]
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_test(stock_data, features, target, n_estimators=100, max_depth=None, random_state=42):
    X, y = stock_data[features], stock_data[target]
    split = 0.5
    
    X_train = X.iloc[:int(X.shape[0] * split)]
    X_test = X.iloc[int(X.shape[0] * split):]
    y_train = y.iloc[:int(y.shape[0] * split)]
    y_test = y.iloc[int(y.shape[0] * split):]
    
    try:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        X_test['y_pred'] = y_pred
        X_test['y_actual'] = y_test
    except Exception as e:
        print(e)

    # print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return X_test
