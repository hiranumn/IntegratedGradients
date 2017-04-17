# IntegratedGradients
Python implementation of integrated gradients (https://arxiv.org/abs/1703.01365). The algorithm "explains" a prediction of a Keras-based deep learning model by approximating Shapley values and assigning them to the input sample features. 

# Usage

Here is a minimal working example on UCI Iris data

```python
from IntegratedGradients import *
from keras.layers import Dense
from keras.layers.core import Activation

X = np.array([[float(j) for j in i.rstrip().split(",")[:-1]] for i in open("iris.data").readlines()][:-1])
Y = np.array([0 for i in range(100)] + [1 for i in range(50)])

model = Sequential([
    Dense(1, input_dim=4),
    Activation('sigmoid'),
])
model.compile(optimizer='sgd', loss='binary_crossentropy')
model.fit(X, Y, epochs=300, batch_size=10, validation_split=0.2, verbose=0)

ig = integrated_gradients(model)
ig.explain(X[0])
==> array([-0.25757075, -0.24014562,  0.12732635,  0.00960122])
```

Email me at hiranumn at cs dot washington dot edu for questions.
