import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import windowshap

features = ['EDA', 'RAW_AF7', 'RAW_AF8', 'RAW_TP9', 'RAW_TP10', 'HeartRate']
sequence_length = 20

df = pd.read_csv('concated_csv/22.csv')
df = df.dropna()
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

X = np.log1p(df[features])
y = df['emotion']
print(np.unique(y))
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

num_classes = len(np.unique(label_encoder.classes_))
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_sequences = np.array([X[i:i + sequence_length] for i in range(len(X) - sequence_length + 1)])
print(X_sequences.shape)
y_one_hot = tf.keras.utils.to_categorical(y_encoded)
y_one_hot = y_one_hot[:len(X_sequences)]

print(y_one_hot)
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_one_hot, test_size=0.4, shuffle=False)


def build_lstm_model(input_shape, num_classes, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile the model
    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    return model


input_shape = (sequence_length, len(features))
model = build_lstm_model(input_shape, num_classes)

# Train the model
#model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test),
#          shuffle=False)

feature_index = 3
X_test_single_feature = X_test[:, :, feature_index]


def model_predict(data):
    return model.predict(data)


def model_predict_idx(data, f_index):
    return model.predict(data)[:, f_index]


# PDP Plot
pdp_values = np.linspace(np.min(X_test_single_feature), np.max(X_test_single_feature), num=100)
pdp_results = []

for value in pdp_values:
    pdp_data = X_test.copy()
    pdp_data[:, :, feature_index] = value
    pdp_results.append(np.mean(model_predict(pdp_data), axis=0))

plt.plot(pdp_values, pdp_results)
plt.xlabel('Feature Values')
plt.ylabel('Model Prediction')
plt.title('Partial Dependence Plot (PDP)')
plt.show()


# Function to calculate the ALE values
def calculate_ale(seq_indices, mdl_predict_idx, f_index):
    ale_values = []
    sorted_indices = np.argsort(seq_indices)

    for i in range(len(seq_indices)):
        predictions = mdl_predict_idx(sorted_indices[:i + 1], f_index)
        ale_value = np.mean(predictions) - np.mean(mdl_predict_idx(sorted_indices[:i], f_index))
        ale_values.append(ale_value)

    return np.array(ale_values)


# Calculate ALE values for the selected feature
sequence_indices = np.arange(len(X_test_single_feature))
calc_ale_values = calculate_ale(sequence_indices, model_predict_idx(X_test_single_feature, 2), feature_index)

# Plot ALE values
plt.plot(sequence_indices, np.cumsum(calc_ale_values))
plt.xlabel('Sequence Indices')
plt.ylabel('Accumulated Local Effects (ALE)')
plt.title('Accumulated Local Effects (ALE) Plot')
plt.show()


loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
print(model.summary())
