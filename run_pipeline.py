from data_preprocessing import load_and_preprocess
from model_training import train_model
from evaluation import evaluate_model

def main():
    # Step 1: Preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess("Indian_Movie_Rating.csv")

    # Step 2: Train
    model = train_model(X_train, y_train)

    # Step 3: Evaluate
    evaluate_model(X_test, y_test)

if __name__ == "__main__":
    main()