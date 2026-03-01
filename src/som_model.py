
import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler


class SOMMicroZone:

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def train_som(self):

        print("Loading dataset...")
        df = pd.read_csv(self.input_path)

        features = df[['temp', 'humidity', 'wind', 'rain', 'elevation']]

        print("Normalizing features...")
        scaler = MinMaxScaler()
        X = scaler.fit_transform(features)

        print("Initializing SOM...")
        som = MiniSom(
            x=3,
            y=3,
            input_len=X.shape[1],
            sigma=1.0,
            learning_rate=0.5,
            random_seed=42
        )

        som.random_weights_init(X)
        print("Training SOM...")
        som.train_random(X, 2000)

        print("Assigning micro-zones...")
        winner_coordinates = np.array([som.winner(x) for x in X])
        micro_zone = winner_coordinates[:, 0] * 3 + winner_coordinates[:, 1]

        df['micro_zone'] = micro_zone

        print("Saving dataset with micro-zones...")
        df.to_csv(self.output_path, index=False)

        print("SOM clustering completed successfully!")

        return df


if __name__ == "__main__":
    som_model = SOMMicroZone(
        input_path="data/processed/master_dataset.csv",
        output_path="data/processed/master_dataset_with_zones.csv"
    )

    df = som_model.train_som()
    print(df[['district', 'micro_zone']].head())