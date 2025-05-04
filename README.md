# 🔧 Custom Dense Layer en TensorFlow/Keras

Este repositorio contiene una implementación desde cero de una **capa densa personalizada** (`CustomDenseLayer`) utilizando la API de subclases de Keras. Esta implementación reproduce el comportamiento de `Dense(units, activation='relu')`, permitiendo máxima flexibilidad y control sobre la arquitectura del modelo.

---

## 📌 ¿Qué se implementa?

La clase `CustomDenseLayer` hereda de `tf.keras.layers.Layer` y define los tres métodos clave:

### 1. `__init__(self, units)`
- Define el número de neuronas (unidades de salida).
- ⚙️ **Parámetros configurables:**
  - `units` *(int)*: número de neuronas en la capa.

### 2. `build(self, input_shape)`
- Crea:
  - `self.w`: matriz de pesos con dimensiones `(input_features, units)`
  - `self.b`: vector de sesgos `(units,)`
- ⚙️ **Flags posibles en `add_weight`:**
  - `initializer='random_normal'` → se puede cambiar por `he_normal`, `glorot_uniform`, etc.
  - `trainable=True` → marcar como entrenable o no

### 3. `call(self, inputs)`
- Define la lógica del paso hacia adelante.
- Aplica producto matricial + sesgo + función de activación:
```python
tf.nn.relu(tf.matmul(inputs, self.w) + self.b)

