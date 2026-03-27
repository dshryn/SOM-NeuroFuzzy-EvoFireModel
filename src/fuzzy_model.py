# src/fuzzy_model.py

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyFireModel:

    def __init__(self, df):
        self.df = df

    def build_fuzzy_system(self, zone_df):

        temp = ctrl.Antecedent(np.arange(0, 51, 1), 'temp')
        humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
        wind = ctrl.Antecedent(np.arange(0, 31, 1), 'wind')
        rain = ctrl.Antecedent(np.arange(0, 101, 1), 'rain')

        risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

        temp_mean = zone_df['temp'].mean()
        humidity_mean = zone_df['humidity'].mean()

        temp['low'] = fuzz.trimf(temp.universe, [0, 0, temp_mean])
        temp['high'] = fuzz.trimf(temp.universe, [temp_mean, 50, 50])

        humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, humidity_mean])
        humidity['high'] = fuzz.trimf(humidity.universe, [humidity_mean, 100, 100])

        wind['low'] = fuzz.trimf(wind.universe, [0, 0, 10])
        wind['high'] = fuzz.trimf(wind.universe, [5, 30, 30])

        rain['low'] = fuzz.trimf(rain.universe, [0, 0, 20])
        rain['high'] = fuzz.trimf(rain.universe, [10, 100, 100])

        risk['low'] = fuzz.trimf(risk.universe, [0, 0, 40])
        risk['medium'] = fuzz.trimf(risk.universe, [30, 50, 70])
        risk['high'] = fuzz.trimf(risk.universe, [60, 100, 100])

        rule1 = ctrl.Rule(temp['high'] & humidity['low'], risk['high'])
        rule2 = ctrl.Rule(wind['high'] & humidity['low'], risk['high'])
        rule3 = ctrl.Rule(rain['high'], risk['low'])
        rule4 = ctrl.Rule(humidity['high'], risk['low'])

        system = ctrl.ControlSystem([rule1, rule2, rule3, rule4])

        return system

    def compute_risk(self):

        all_results = []

        for zone in self.df['zone'].unique():

            zone_df = self.df[self.df['zone'] == zone]
            system = self.build_fuzzy_system(zone_df)

            for idx, row in zone_df.iterrows():

                sim = ctrl.ControlSystemSimulation(system)

                sim.input['temp'] = row['temp']
                sim.input['humidity'] = row['humidity']
                sim.input['wind'] = row['wind']
                sim.input['rain'] = row['rain']

                sim.compute()

                risk_value = sim.output.get('risk', 0)
                all_results.append((idx, risk_value))

        all_results.sort(key=lambda x: x[0])
        risks = [r[1] for r in all_results]

        self.df['fuzzy_risk'] = risks

        return self.df