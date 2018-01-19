# SimpleFlow
A simple TensorFlow-like graph computation framework in Python for learning purpose

## Example
``` python
import simpleflow as sf

# Create a graph
with sf.Graph().as_default():
    a = sf.Variable(1.0, name='a')
    b = sf.Variable(2.0, name='b')
    result = sf.add(a, b, name='a+b')

    # Create a session to run the graph 
    with sf.Session() as sess:
        print(sess.run(result))
```

## TODO List

- [x] Computational Graph
- [x] Feed forward propagation
- [ ] Backpropagation
- [ ] GradientDescent Optimizer

