# model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, model_from_json

def build_autoencoder(input_dim: int, bottleneck: int = 50):
    tf.keras.backend.clear_session()
    input_data = Input(shape=(input_dim,))
    encoder = Dense(bottleneck, activation='linear')(input_data)
    decoder = Dense(input_dim, activation='linear')(encoder)
    autoencoder = Model(input_data, decoder)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=opt, loss='mse')
    encoder_model = Model(input_data, encoder)
    return autoencoder, encoder_model


def save_autoencoder(model, filename):
    model_json = model.to_json()
    with open(filename + ".json", "w") as json_file:
      json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename + ".weights.h5")
    print("Saved model to disk")


def load_autoencoder(filename):
    with open(filename + '.json', 'r') as f:
        loaded_model_json = f.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename + ".weights.h5")
    print("Loaded model from disk")
    
    return loaded_model