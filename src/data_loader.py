import pandas as pd
import glob
import os


class DataLoader:

    def __init__(self, fires_path, weather_path, output_path):
        self.fires_path = fires_path
        self.weather_path = weather_path
        self.output_path = output_path

        self.elevation_map = {
            "Almora": 1650,
            "Dehradun": 640,
            "Haridwar": 314,
            "Nainital": 2084
        }

    def load_fire_data(self):

        print("Loading fire Excel files...")

        files = glob.glob(os.path.join(self.fires_path, "*.xlsx"))

        if len(files) == 0:
            raise ValueError("No fire files found.")

        df_list = []

        for file in files:
            df = pd.read_excel(file)
            df_list.append(df)

        fires = pd.concat(df_list, ignore_index=True)

        fires['acq_date'] = pd.to_datetime(
            fires['acq_date'], errors='coerce'
        ).dt.tz_localize(None)

        fires = fires.dropna(subset=['acq_date'])

        fires = fires[['latitude', 'longitude', 'acq_date', 'confidence']]

        print("Fire data shape:", fires.shape)

        return fires

    # ---------------------------------------------------
    # Assign district by coordinates
    # ---------------------------------------------------
    def assign_district(self, lat, lon):

        if 29.5 <= lat <= 30.5 and 79 <= lon <= 80:
            return "Almora"
        elif 30 <= lat <= 31 and 77.8 <= lon <= 78.5:
            return "Dehradun"
        elif 29.8 <= lat <= 30.5 and 78 <= lon <= 79:
            return "Haridwar"
        elif 29 <= lat <= 29.7 and 79.3 <= lon <= 80:
            return "Nainital"
        else:
            return None

    # ---------------------------------------------------
    # Create daily fire indicator
    # ---------------------------------------------------
    def create_fire_indicator(self, fires):

        fires['district'] = fires.apply(
            lambda x: self.assign_district(x['latitude'], x['longitude']),
            axis=1
        )

        fires = fires.dropna(subset=['district'])

        fire_daily = (
            fires.groupby(['acq_date', 'district'])
            .size()
            .reset_index(name='fire_count')
        )

        fire_daily['fire_occurred'] = 1

        print("Fire daily shape:", fire_daily.shape)

        return fire_daily[['acq_date', 'district', 'fire_occurred']]

    # ---------------------------------------------------
    # LOAD WEATHER DATA
    # ---------------------------------------------------
    def load_weather_data(self):

        print("Loading weather Excel files...")

        files = glob.glob(os.path.join(self.weather_path, "*.xlsx"))

        if len(files) == 0:
            raise ValueError("No weather files found.")

        df_list = []

        for file in files:

            district_name = os.path.basename(file).split(".")[0].strip()

            # Ensure standardized naming
            district_name = district_name.title()

            df = pd.read_excel(file)

            df['date'] = pd.to_datetime(
                df['date'], errors='coerce'
            ).dt.tz_localize(None)

            df = df.dropna(subset=['date'])

            df = df.rename(columns={
                'temperature_2m': 'temp',
                'relative_humidity_2m': 'humidity',
                'wind_speed_10m': 'wind'
            })

            # Ensure rain column exists
            if 'rain' not in df.columns:
                if 'precipitation' in df.columns:
                    df['rain'] = df['precipitation']
                else:
                    df['rain'] = 0

            df['district'] = district_name

            df = df[['date', 'district', 'temp', 'humidity', 'rain', 'wind']]

            df_list.append(df)

        weather = pd.concat(df_list, ignore_index=True)

        print("Weather data shape:", weather.shape)

        return weather

    # ---------------------------------------------------
    # MERGE EVERYTHING
    # ---------------------------------------------------
    def merge_datasets(self):

        fires = self.load_fire_data()
        fire_daily = self.create_fire_indicator(fires)
        weather = self.load_weather_data()

        fire_daily = fire_daily.rename(columns={'acq_date': 'date'})

        master_df = weather.merge(
            fire_daily,
            on=['date', 'district'],
            how='left'
        )

        master_df['fire_occurred'] = master_df['fire_occurred'].fillna(0)

        # Elevation mapping
        master_df['elevation'] = master_df['district'].map(self.elevation_map)

        # Clean rows with missing weather only
        master_df = master_df.dropna(subset=['temp', 'humidity', 'wind'])

        master_df['rain'] = master_df['rain'].fillna(0)

        # Derived feature
        master_df['dryness_index'] = (
            master_df['temp'] * (100 - master_df['humidity']) / 100
        )

        master_df = master_df.sort_values(['date', 'district'])

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        master_df.to_csv(self.output_path, index=False)

        print("\nMaster dataset created successfully.")
        print("Final shape:", master_df.shape)

        return master_df


if __name__ == "__main__":

    loader = DataLoader(
        fires_path="data/raw/fires/",
        weather_path="data/raw/weather/",
        output_path="data/processed/master_dataset.csv"
    )

    df = loader.merge_datasets()
    print(df.head())