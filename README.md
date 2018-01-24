## *SimpleFlow*
A simple TensorFlow-like graph computation framework in Python for learning purpose

### Quick Start
``` python
import simpleflow as sf

# Create a graph
with sf.Graph().as_default():
    a = sf.constant(1.0, name='a')
    b = sf.constant(2.0, name='b')
    result = sf.add(a, b, name='a+b')

    # Create a session to run the graph 
    with sf.Session() as sess:
        print(sess.run(result))
```

### Examples
- [Feedforward](https://github.com/PytLab/simpleflow/blob/master/exmaples/feedforward.ipynb)
- [Linear Regression](https://github.com/PytLab/simpleflow/blob/master/exmaples/linear_regression.ipynb)

### Features

- [x] Computational Graph
- [x] Feed forward propagation
- [x] Backpropagation
- [x] GradientDescent Optimizer
- [x] Linear Regression Example
- [ ] MNIST classification Example

