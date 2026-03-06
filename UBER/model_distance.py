import numpy as np

lat1, lon1 = np.radians(df['pickup_latitude']), np.radians(df['pickup_longitude'])
lat2, lon2 = np.radians(df['dropoff_latitude']), np.radians(df['dropoff_longitude'])

dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2

c = 2 * np.arcsin(np.sqrt(a))
df['distance_km'] = 6371 * c
print(df['distance_km'])

X = df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']] 
y = df['distance_km']
df = df.dropna()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
