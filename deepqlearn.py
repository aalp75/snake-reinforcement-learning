import os
import tensorflow as tf

class DeepQlearn:
    """
    Deep Q-learning model using a feed-forward neural network

    The network takes the 11-dimensional state representation as input
    and outputs Q-values for the three possible actions:
        - straight
        - right
        - left
    """
    def __init__(self, learning_rate=0.001):

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, input_shape=(11,), activation="relu"),
            tf.keras.layers.Dense(units=128, activation ="relu"),
            tf.keras.layers.Dense(units=3) # Q-values for the 3 actions
        ])
    
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')
        
        self.model.summary()
        
    def predict(self, x):
        #return self.model.predict(x, verbose=0)
        return self.model(x, training=False).numpy()
        
    def save(self):
        os.makedirs("parameters", exist_ok=True)
        print("-- Config saved --")
        self.model.save_weights("parameters/deepqlearn_cfg.weights.h5")

    def load(self):
        print("-- Config loaded --")
        self.model.load_weights("parameters/deepqlearn_cfg.weights.h5")



