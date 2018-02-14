# Integrated Gradients
Python implementation of integrated gradients [1]. The algorithm "explains" a prediction of a Keras-based deep learning model by approximating Aumann–Shapley values for the input features. These values allocate the difference between the model prediction for a reference value (all zeros by default) and the prediction for the current sample among the input features. **[TensorFlow version](https://github.com/hiranumn/IntegratedGradientsTF) is implemented now!**

# Usage

Using Integrated_Gradients is very easy. There is no need to modify your Keras model.  
Here is a minimal working example on UCI Iris data.

1. Build your own Keras model and train it. Make sure to complie it!
``` Python
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
```

2. Wrap it with an integrated_gradients instance.
``` Python
ig = integrated_gradients(model)
```

3. Call explain() with a sample to explain.
``` Python
ig.explain(X[0])
==> array([-0.25757075, -0.24014562,  0.12732635,  0.00960122])
```

# Features
- supports both Sequential() and Model() instances.
- supports both **TensorFlow** and **Theano** backends.
- works on models with multiple outputs.
- works on models with mulitple input branches.

# Example notebooks
- More thorough example can be found [here](examples/example.ipynb).  
- There is also an [example](examples/VGG%20example.ipynb) of running this on VGG16 model.  
- If your network has multiple input sources (branches), you can take a look at [this](examples/Networks%20with%20multiple%20inputs.ipynb). 

# MNIST example
We trained a simple CNN model (1 conv layer and 1 dense layer) on the MNIST imagesets. 
Here are some results of running integrated_gradients on the trained model and explaining some samples.

![alt text](notebooks/figures/13206.png)
![alt text](notebooks/figures/13254.png)
![alt text](notebooks/figures/14335.png)
![alt text](notebooks/figures/16328.png)
![alt text](notebooks/figures/18745.png)
![alt text](notebooks/figures/1995.png)
![alt text](notebooks/figures/23525.png)

# References
1. Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. "Axiomatic Attribution for Deep Networks." arXiv preprint arXiv:1703.01365 (2017).

Email me at hiranumn at cs dot washington dot edu for questions.
