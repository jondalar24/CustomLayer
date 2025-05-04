# custom_dense_model.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Softmax, Dropout
from tensorflow.keras.models import Sequential

# Capa densa personalizada con activación ReLU
class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CustomDenseLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)

# Modelo secuencial con dos capas personalizadas y Softmax
model = Sequential([
    CustomDenseLayer(128),
    Dropout(0.5),                # Previene sobreajuste
    CustomDenseLayer(10),
    Softmax()                    # Para clasificación multiclase
])

#  Compilación del modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Generación de datos sintéticos (puedes sustituir por dataset real)
x_train = np.random.random((1000, 20))
y_train = np.random.randint(10, size=(1000, 1))
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

x_test = np.random.random((200, 20))
y_test = np.random.randint(10, size=(200, 1))
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Entrenamiento del modelo
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluación
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nTest loss: {loss:.4f}")
print(f"Test accuracy: {accuracy:.4f}")
