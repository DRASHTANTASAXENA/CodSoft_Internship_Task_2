from sklearn.linear_model import LinearRegression
import joblib

def train_model(X_train, y_train, save_path="linear_model.pkl"):
    LR = LinearRegression()
    LR.fit(X_train, y_train)
    joblib.dump(LR, save_path)
    return LR