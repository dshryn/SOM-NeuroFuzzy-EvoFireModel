from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import pandas as pd

df = pd.read_csv("data/processed/master_dataset.csv")

df['date'] = pd.to_datetime(df['date'])

train = df[df['date'] < "2022-01-01"]
test  = df[df['date'] >= "2022-01-01"]

print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\nTrain fire rate:", train['fire_occurred'].mean())
print("Test fire rate:", test['fire_occurred'].mean())

features = ['temp', 'humidity', 'wind', 'rain', 'dryness_index', 'elevation']

X_train = train[features]
y_train = train['fire_occurred']

X_test = test[features]
y_test = test['fire_occurred']

model = LogisticRegression(
    class_weight={0:1, 1:100},
    max_iter=1000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))