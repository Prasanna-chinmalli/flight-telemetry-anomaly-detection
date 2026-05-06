import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000

data = {
    "time": pd.date_range(start="2025-01-01", periods=n, freq="s"),
    "altitude": np.random.normal(35000, 500, n),
    "speed": np.random.normal(850, 30, n),
    "engine_temperature": np.random.normal(650, 20, n),
    "fuel_flow": np.random.normal(2500, 150, n),
    "vibration": np.random.normal(0.5, 0.1, n),
}

df = pd.DataFrame(data)

anomaly_indices = np.random.choice(n, 30, replace=False)

df.loc[anomaly_indices, "engine_temperature"] += np.random.normal(180, 30, 30)
df.loc[anomaly_indices, "vibration"] += np.random.normal(1.2, 0.2, 30)
df.loc[anomaly_indices, "fuel_flow"] += np.random.normal(700, 100, 30)

df.to_csv("flight_data.csv", index=False)

print("flight_data.csv generated successfully!")