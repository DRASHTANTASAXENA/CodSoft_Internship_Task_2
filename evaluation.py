import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import joblib

def evaluate_model(X_test, y_test, model_path="linear_model.pkl"):
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # R² score
    r2 = r2_score(y_test, y_pred) * 100
    print("R² Score:", r2)

    # Results dataframe
    results = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred.ravel(),
        'residual': y_test - y_pred
    })
    print(results.head())

    # Scatterplot
    sns.scatterplot(x=results['actual'], y=results['predicted'])
    plt.title('Actual vs Predicted values')
    plt.show()

    return r2, results