from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_KNN_model(stock_data, features, target, model=KNeighborsClassifier()):
    X_train, y_train = stock_data[features], stock_data[target]
    model.fit(X_train, y_train)
    return model

def train_random_forest_model(stock_data, features, target, n_estimators=100, max_depth=None, random_state=42):
    X_train, y_train = stock_data[features], stock_data[target]
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_test(stock_data, features, target, n_estimators=100, max_depth=None, random_state=42):
    X, y = stock_data[features], stock_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
