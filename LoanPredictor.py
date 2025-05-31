import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data
data = {
    'Income': [30, 50, 70, 20, 90],    # in $1000s
    'Debt':   [10, 20, 25, 15, 5],     # in $1000s
    'Approved': [0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
X = df[['Income', 'Debt']].values
y = df['Approved'].values

# Define model
model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, batch_size=1, verbose=0)

# Predict on a new example
test_data = np.array([[60, 10]])
prediction = model.predict(test_data)
predicted_label = (prediction > 0.5).astype(int)

print(f"Prediction: {'Approved' if predicted_label[0][0] == 1 else 'Denied'} (Confidence: {prediction[0][0]:.2f})")
