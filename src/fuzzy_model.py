# src/fuzzy_model.py

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyFireModel:

    def __init__(self, df):
        self.df = df

        # Precompute fire statistics (IMPORTANT for imbalance handling)
        self.fire_df = df[df['fire_occurred'] == 1]

        if len(self.fire_df) > 0:
            self.fire_temp_mean = self.fire_df['temp'].mean()
            self.fire_humidity_mean = self.fire_df['humidity'].mean()
        else:
            self.fire_temp_mean = df['temp'].mean()
            self.fire_humidity_mean = df['humidity'].mean()

    def build_fuzzy_system(self, zone_df):

        # -----------------------------
        # Define fuzzy variables
        # -----------------------------
        temp = ctrl.Antecedent(np.arange(0, 51, 1), 'temp')
        humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
        wind = ctrl.Antecedent(np.arange(0, 31, 1), 'wind')
        rain = ctrl.Antecedent(np.arange(0, 101, 1), 'rain')

        risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

        # -----------------------------
        # Zone + Fire adaptive centers
        # -----------------------------
        zone_temp_mean = zone_df['temp'].mean()
        zone_humidity_mean = zone_df['humidity'].mean()

        # Combine zone + fire behavior
        temp_high_center = (zone_temp_mean + self.fire_temp_mean) / 2
        humidity_low_center = self.fire_humidity_mean

        # -----------------------------
        # Membership functions
        # -----------------------------
        temp['low'] = fuzz.trimf(temp.universe, [0, 0, temp_high_center])
        temp['high'] = fuzz.trimf(temp.universe, [temp_high_center, 50, 50])

        humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, humidity_low_center])
        humidity['high'] = fuzz.trimf(humidity.universe, [humidity_low_center, 100, 100])

        wind['low'] = fuzz.trimf(wind.universe, [0, 0, 10])
        wind['high'] = fuzz.trimf(wind.universe, [5, 30, 30])

        rain['low'] = fuzz.trimf(rain.universe, [0, 0, 20])
        rain['high'] = fuzz.trimf(rain.universe, [10, 100, 100])

        # 🔥 Make high-risk sharper (important)
        risk['low'] = fuzz.trimf(risk.universe, [0, 0, 30])
        risk['medium'] = fuzz.trimf(risk.universe, [25, 50, 75])
        risk['high'] = fuzz.trimf(risk.universe, [50, 90, 100])

        # -----------------------------
        # RULE BASE (IMPROVED)
        # -----------------------------

        # Strong fire conditions
        rule1 = ctrl.Rule(temp['high'] & humidity['low'], risk['high'])
        rule2 = ctrl.Rule(wind['high'] & humidity['low'], risk['high'])
        rule5 = ctrl.Rule(temp['high'] & humidity['low'] & wind['high'], risk['high'])

        # Rain reduces fire risk
        rule3 = ctrl.Rule(rain['high'], risk['low'])

        # Weakened low-risk rule (IMPORTANT FIX)
        rule4 = ctrl.Rule(humidity['high'] & temp['low'], risk['low'])

        # Optional balancing rule
        rule6 = ctrl.Rule(temp['high'] & humidity['high'], risk['medium'])

        system = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, rule6
        ])

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

        # Maintain order
        all_results.sort(key=lambda x: x[0])
        risks = [r[1] for r in all_results]

        self.df['fuzzy_risk'] = risks

        return self.df