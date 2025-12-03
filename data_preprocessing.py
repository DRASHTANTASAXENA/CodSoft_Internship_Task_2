import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess(path="Indian_Movie_Rating.csv"):
    # Load dataset
    data = pd.read_csv(path, encoding="latin 1")

    # Drop missing values
    data.dropna(inplace=True)

    # Clean Duration column
    data['Duration'] = pd.to_numeric(data['Duration'].str.strip('min'))
    data['Duration'].fillna(data['Duration'].mean(), inplace=True)

    # Drop duplicates
    data.drop_duplicates(inplace=True)

    # Fix Year column
    year = []
    for y in data.Year:
        if isinstance(y, float):
            year.append(np.nan)
        else:
            year.append(int(str(y)[1:5]))
    data["Year"] = year

    # Fix Votes column
    data["Votes"] = data["Votes"].replace("$5.16M", 516)
    data["Votes"] = pd.to_numeric(data['Votes'].str.replace(',', ''))

    # Feature encoding
    data_update = data.drop(['Name'], axis=1)
    actor1_encoding_map = data_update.groupby('Actor 1').agg({'Rating': 'mean'}).to_dict()
    actor2_encoding_map = data_update.groupby('Actor 2').agg({'Rating': 'mean'}).to_dict()
    actor3_encoding_map = data_update.groupby('Actor 3').agg({'Rating': 'mean'}).to_dict()
    director_encoding_map = data_update.groupby('Director').agg({'Rating': 'mean'}).to_dict()
    genre_encoding_map = data_update.groupby('Genre').agg({'Rating': 'mean'}).to_dict()

    data_update['actor1_encoded'] = round(data_update['Actor 1'].map(actor1_encoding_map['Rating']), 1)
    data_update['actor2_encoded'] = round(data_update['Actor 2'].map(actor2_encoding_map['Rating']), 1)
    data_update['actor3_encoded'] = round(data_update['Actor 3'].map(actor3_encoding_map['Rating']), 1)
    data_update['director_encoded'] = round(data_update['Director'].map(director_encoding_map['Rating']), 1)
    data_update['genre_encoded'] = round(data_update['Genre'].map(genre_encoding_map['Rating']), 1)

    data_update.drop(['Actor 1', 'Actor 2', 'Actor 3', 'Director', 'Genre'], axis=1, inplace=True)

    # Split features and target
    X = data_update.drop('Rating', axis=1)
    y = data_update['Rating']

    return train_test_split(X, y, test_size=0.2, random_state=42)