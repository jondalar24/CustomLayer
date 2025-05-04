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
```
---
### ✅ 2. Construcción del modelo
```python
model = Sequential([
    CustomDenseLayer(128),
    Dropout(0.5),
    CustomDenseLayer(10),
    Softmax()
])
```
#### Se compone de dos capas densas personalizadas:

- 128 neuronas ocultas con ReLU
- 10 neuronas para salida multiclase
- Dropout(0.5) ayuda a prevenir el sobreajuste
- Softmax() convierte las salidas en probabilidades (clasificación multiclase)
---
### ✅ 3. Compilación del modelo
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

* adam: optimizador adaptativo muy usado
* categorical_crossentropy: apropiado para clasificación multiclase con etiquetas one-hot
---
### ✅ 4. Datos sintéticos para entrenamiento
```python
x_train = np.random.random((1000, 20))
y_train = np.random.randint(10, size=(1000, 1))
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

x_train: 1000 muestras con 20 variables de entrada

y_train: 10 clases posibles (0 a 9), convertidas a one-hot
```
#### 📌 Puedes reemplazar estos datos por un dataset real para observar un aprendizaje significativo.
---
### ✅ 5. Entrenamiento y evaluación
```python
model.fit(x_train, y_train, epochs=10, batch_size=32)
model.evaluate(x_test, y_test)
```
- Entrena durante 10 épocas con batches de 32 muestras
- Evalúa la pérdida y precisión sobre un conjunto de test también aleatorio
---
## ⚙️ Parámetros que puedes modificar

| Parámetro       | Lugar en el código          | Valores posibles / efecto                                       |
|-----------------|-----------------------------|------------------------------------------------------------------|
| `units`         | `CustomDenseLayer(units)`   | Número de neuronas por capa                                     |
| `activation`    | En `call()`                 | `tf.nn.relu`, `tanh`, `sigmoid`, etc.                           |
| `initializer`   | En `add_weight()`           | `'he_normal'`, `'glorot_uniform'`, `'random_normal'`, etc.      |
| `Dropout rate`  | `Dropout(0.5)`              | Valor entre 0 y 1                                                |
| `loss`          | `model.compile()`           | `'categorical_crossentropy'`, `'sparse_categorical_crossentropy'` |
| `optimizer`     | `adam`, `sgd`, `rmsprop`, etc. | Afecta cómo se ajustan los pesos durante el entrenamiento     |

---
#### 🧪 Ejecución
Este script puede ejecutarse directamente en un entorno con TensorFlow 2.x:
```bash
python custom_dense_model.py
```
---
### 📚 Aplicaciones
Esta técnica de crear capas personalizadas es útil cuando:

- Quieres experimentar con nuevas arquitecturas
- Necesitas añadir lógica específica no disponible en las capas estándar
- Estás trabajando en investigación o pruebas de concepto

## 🧠 Créditos y contexto
Esta práctica tiene como objetivo afianzar el uso de la API de subclases para crear capas reutilizables, entendiendo a fondo cómo funcionan internamente las redes neuronales.
