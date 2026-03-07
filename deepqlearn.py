import tensorflow as tf

class DeepQlearn:
    
    def __init__(self, learning_rate=0.001):

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, input_dim=11, activation ="relu"),
            tf.keras.layers.Dense(units=128, activation ="relu"),
            tf.keras.layers.Dense(units=3)
        ])
    
        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate), loss='mse')
        
        self.model.summary()
        
    def predict(self, x):
        return self.model.predict(x)
        
    def save(self):
        print("----Config saved----")
        self.model.save_weights('save_parameters/deepqlearn_cfg')
        
    def load(self):
        print("----Config loaded----")
        self.model.load_weights('save_parameters/deepqlearn_cfg')



