# src/som_model.py

import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class SOMModel:

    def __init__(self, x=3, y=3, input_len=5, sigma=1.0, learning_rate=0.5):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.som = None
        self.scaler = MinMaxScaler()

    def fit(self, df, features):

        data = df[features].values
        data_scaled = self.scaler.fit_transform(data)

        self.som = MiniSom(
            self.x,
            self.y,
            self.input_len,
            sigma=self.sigma,
            learning_rate=self.learning_rate
        )

        self.som.random_weights_init(data_scaled)
        self.som.train_random(data_scaled, 1500)

        print("SOM training complete.")

        labels = []
        for row in data_scaled:
            winner = self.som.winner(row)
            zone_label = winner[0] * self.y + winner[1]
            labels.append(zone_label)

        df['zone'] = labels

        print("Zones created:", df['zone'].nunique())

        return df

    def plot_som_grid(self, df, features):

        data = self.scaler.transform(df[features].values)

        plt.figure(figsize=(7,7))
        plt.pcolor(self.som.distance_map().T, cmap='coolwarm')
        plt.colorbar()

        for i, x in enumerate(data):
            w = self.som.winner(x)
            plt.text(w[0]+0.5, w[1]+0.5, '.', color='black')

        plt.title("SOM Distance Map (U-Matrix)")
        plt.show()