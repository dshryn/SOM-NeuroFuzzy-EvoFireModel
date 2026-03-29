import pandas as pd
import os


def create_balanced_dataset(input_path, output_path):

    print("Loading dataset...")
    df = pd.read_csv(input_path)

    print("\nOriginal distribution:")
    print(df['fire_occurred'].value_counts())

    # Separate classes
    fire_df = df[df['fire_occurred'] == 1]
    non_fire_df = df[df['fire_occurred'] == 0]

    print("\nFire count:", len(fire_df))
    print("Non-fire count:", len(non_fire_df))

    # Undersample non-fire
    non_fire_sample = non_fire_df.sample(
        n=len(fire_df),
        random_state=42
    )

    # Combine
    df_balanced = pd.concat([fire_df, non_fire_sample])

    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nBalanced distribution:")
    print(df_balanced['fire_occurred'].value_counts())

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_balanced.to_csv(output_path, index=False)

    print("\nBalanced dataset saved to:", output_path)
    print("Shape:", df_balanced.shape)


if __name__ == "__main__":

    create_balanced_dataset(
        input_path="data/processed/master_dataset.csv",
        output_path="data/processed/balanced_dataset.csv"
    )