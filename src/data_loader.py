import pandas as pd
import glob
import os


class DataLoader:

    def __init__(self, modis_path, weather_path, output_path):
        self.modis_path = modis_path
        self.weather_path = weather_path
        self.output_path = output_path

        # static
        self.elevation_map = {
            "Almora": 1650,
            "Dehradun": 640,
            "Haridwar": 314,
            "Nainital": 2084
        }

    def load_modis_data(self):
        print("Loading MODIS files...")
        files = glob.glob(os.path.join(self.modis_path, "*.csv"))
        df_list = [pd.read_csv(file) for file in files]
        modis = pd.concat(df_list, ignore_index=True)

        modis['acq_date'] = pd.to_datetime(modis['acq_date'])

        return modis

    def assign_district(self, lat, lon):
        # approx
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

    def create_fire_indicator(self, modis):
        print("Assigning districts to fire points...")
        modis['district'] = modis.apply(
            lambda x: self.assign_district(x['latitude'], x['longitude']), axis=1
        )

        modis = modis.dropna(subset=['district'])

        fire_daily = (
            modis.groupby(['acq_date', 'district'])
            .size()
            .reset_index(name='fire_count')
        )

        fire_daily['fire_occurred'] = 1

        return fire_daily[['acq_date', 'district', 'fire_occurred']]

    def load_weather_data(self):
        print("Loading weather data...")
        weather = pd.read_csv(self.weather_path)

        weather['date'] = pd.to_datetime(weather['date'])

        weather = weather[[
            'date',
            'district',
            'temperature_2m',
            'relative_humidity_2m',
            'precipitation',
            'wind_speed_10m'
        ]]

        weather = weather.rename(columns={
            'temperature_2m': 'temp',
            'relative_humidity_2m': 'humidity',
            'precipitation': 'rain',
            'wind_speed_10m': 'wind'
        })

        return weather

    def merge_datasets(self):
        modis = self.load_modis_data()
        fire_daily = self.create_fire_indicator(modis)
        weather = self.load_weather_data()

        print("Creating full date-district grid...")

        all_dates = pd.date_range(
            start=weather['date'].min(),
            end=weather['date'].max()
        )

        districts = list(self.elevation_map.keys())

        full_index = pd.MultiIndex.from_product(
            [all_dates, districts],
            names=['date', 'district']
        )

        fire_full = pd.DataFrame(index=full_index).reset_index()

        fire_full = fire_full.merge(
            fire_daily,
            left_on=['date', 'district'],
            right_on=['acq_date', 'district'],
            how='left'
        )

        fire_full['fire_occurred'] = fire_full['fire_occurred'].fillna(0)
        fire_full = fire_full.drop(columns=['acq_date'])

        print("Merging weather and fire data...")
        master_df = weather.merge(
            fire_full,
            on=['date', 'district'],
            how='left'
        )

        master_df['fire_occurred'] = master_df['fire_occurred'].fillna(0)

        print("Adding elevation...")
        master_df['elevation'] = master_df['district'].map(self.elevation_map)

        master_df = master_df.dropna()

        print("Saving master dataset...")
        master_df.to_csv(self.output_path, index=False)

        print("Master dataset saved successfully!")

        return master_df


if __name__ == "__main__":
    loader = DataLoader(
        modis_path="data/raw/modis/",
        weather_path="data/raw/weather.csv",
        output_path="data/processed/master_dataset.csv"
    )

    master_df = loader.merge_datasets()
    print(master_df.head())