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

### Blogs
- [实现属于自己的TensorFlow(一) - 计算图与前向传播](http://pytlab.github.io/2018/01/24/%E5%AE%9E%E7%8E%B0%E5%B1%9E%E4%BA%8E%E8%87%AA%E5%B7%B1%E7%9A%84TensorFlow-%E4%B8%80-%E8%AE%A1%E7%AE%97%E5%9B%BE%E4%B8%8E%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD/)
- [实现属于自己的TensorFlow(二) - 梯度计算与反向传播](http://pytlab.github.io/2018/01/25/%E5%AE%9E%E7%8E%B0%E5%B1%9E%E4%BA%8E%E8%87%AA%E5%B7%B1%E7%9A%84TensorFlow-%E4%BA%8C-%E6%A2%AF%E5%BA%A6%E8%AE%A1%E7%AE%97%E4%B8%8E%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD/)
- [实现属于自己的TensorFlow(三) - 反向传播与梯度下降实现](http://pytlab.github.io/2018/01/27/%E5%AE%9E%E7%8E%B0%E5%B1%9E%E4%BA%8E%E8%87%AA%E5%B7%B1%E7%9A%84TensorFlow-%E4%B8%89-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E4%B8%8E%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%AE%97%E6%B3%95%E5%AE%9E%E7%8E%B0/)

