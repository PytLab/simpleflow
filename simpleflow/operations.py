#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Operation classes in computational graph.
'''
from queue import Queue

import numpy as np

class Operation(object):
    ''' Base class for all operations in simpleflow.

    An operation is a node in computational graph receiving zero or more nodes
    as input and produce zero or more nodes as output. Vertices could be an
    operation, variable or placeholder.
    '''
    def __init__(self, *input_nodes, name=None):
        ''' Operation constructor.

        :param input_nodes: Input nodes for the operation node.
        :type input_nodes: Objects of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        # Nodes received by this operation.
        self.input_nodes = input_nodes

        # Nodes that receive this operation node as input.
        self.output_nodes = []

        # Output value of this operation in session execution.
        self.output_value = None

        # Operation name.
        self.name = name

        # Graph the operation belongs to.
        self.graph = DEFAULT_GRAPH

        # Add this operation node to destination lists in its input nodes.
        for node in input_nodes:
            node.output_nodes.append(self)

        # Add this operation to default graph.
        self.graph.operations.append(self)

    def compute_output(self):
        ''' Compute and return the output value of the operation.
        '''
        raise NotImplementedError

    def compute_gradient(self, grad=None):
        ''' Compute and return the gradient of the operation wrt inputs.
        '''
        raise NotImplementedError

# ------------------------------------------------------------------------------
# Addition operation
# ------------------------------------------------------------------------------

class Add(Operation):
    ''' An addition operation.
    '''
    def __init__(self, x, y, name=None):
        ''' Addition constructor.

        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        ''' Compute and return the value of addition operation.
        '''
        x, y = self.input_nodes
        self.output_value = np.add(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        ''' Compute the gradients for this operation wrt input values.

        :param grad: The gradient of other operation wrt the addition output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        x, y = [node.output_value for node in self.input_nodes]
        if x.shape != y.shape:
            raise ValueError('Input shapes must be equal for add operation')

        if grad is None:
            grad = np.ones_like(self.output_value)

        return [1.0*grad, 1.0*grad]

def add(x, y, name=None):
    ''' Returns x + y element-wise.
    '''
    return Add(x, y, name)

# ------------------------------------------------------------------------------
# Multiplication operation
# ------------------------------------------------------------------------------

class Multiply(Operation):
    ''' Multiplication operation.
    '''
    def __init__(self, x, y, name=None):
        ''' Multiplication constructor.

        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        ''' Compute and return the multiplication operation result.
        '''
        x, y = self.input_nodes
        self.output_value = np.multiply(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        ''' Compute and return gradients for this operation wrt input values.

        :param grad: The gradient of other operation wrt the mutiply output.
        :type grad: number or a ndarray.
        '''
        x, y = [node.output_value for node in self.input_nodes]
        if x.shape != y.shape:
            raise ValueError('Input shapes must be equal for multiplication operation')

        if grad is None:
            grad = np.ones_like(self.output_value)

        return [np.multiply(y, grad), np.multiply(x, grad)]

def multiply(x, y, name=None):
    ''' Returns x * y element-wise.
    '''
    return Multiply(x, y, name)

# ------------------------------------------------------------------------------
# Matrix multiplication operation
# ------------------------------------------------------------------------------

class MatMul(Operation):
    ''' Matrix multiplication operation.
    '''
    def __init__(self, x, y, name=None):
        ''' MatMul constructor.

        :param x: The first input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param y: The second input node.
        :type y: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        ''' Compute and return the multiplication operation result.
        '''
        x, y = self.input_nodes
        self.output_value = np.dot(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        ''' Compute and return the gradient for matrix multiplication.

        :param grad: The gradient of other operation wrt the matmul output.
        :type grad: number or a ndarray, default value is 1.0.
        '''
        # Get input values.
        x, y = [node.output_value for node in self.input_nodes]

        # Default gradient wrt the matmul output.
        if grad is None:
            grad = np.ones_like(self.output_value)

        # Gradients wrt inputs.
        dfdx = np.dot(grad, y.T)
        dfdy = np.dot(x.T, grad)

        return [dfdx, dfdy]

def matmul(x, y, name=None):
    ''' Multiplies matrix `a` by matrix `b`, producing `a` * `b`.
    '''
    return MatMul(x, y, name)


class Sigmoid(Operation):
    ''' Sigmoid operation.
    '''
    def __init__(self, x, name=None):
        ''' Sigmoid operation constructor.

        :param x: The input node.
        :type x: Object of `Operation`, `Variable` or `Placeholder`.

        :param name: The operation name.
        :type name: str.
        '''
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        ''' Compute and return the value of sigmoid function.
        '''
        x, = self.input_nodes
        self.output_value = 1/(1 + np.exp(-x.output_value))
        return self.output_value

    def compute_gradient(self, grad=None):
        ''' Compute the gradient for sigmoid operation wrt input value.

        :param grad: The gradient of other operation wrt the sigmoid output.
        :type grad: ndarray.
        '''
        if grad is None:
            grad = np.ones_like(self.output_value)
        return grad*self.output_valuea*(1 - self.output_value)

# ------------------------------------------------------------------------------
# Constant node
# ------------------------------------------------------------------------------

class Constant(object):
    ''' Constant node in computational graph.
    '''
    def __init__(self, value, name=None):
        ''' Cosntant constructor.
        '''
        # Constant value.
        if np.shape(value):
            self.value = np.array(value)
        else:
            self.value = np.array([value])

        # Output value of this operation in session.
        self.output_value = None

        # Nodes that receive this variable node as input.
        self.output_nodes = []

        # Operation name.
        self.name = name

        # Add to graph.
        DEFAULT_GRAPH.constants.append(self)

    def compute_output(self):
        ''' Compute and return the constant value.
        '''
        if self.output_value is None:
            self.output_value = self.value
        return self.output_value

def constant(value, name=None):
    ''' Create a constant node.
    '''
    return Constant(value, name=name)

# ------------------------------------------------------------------------------
# Variable node
# ------------------------------------------------------------------------------

class Variable(object):
    ''' Variable node in computational graph.
    '''
    def __init__(self, initial_value=None, name=None, trainable=True): 
        ''' Variable constructor.

        :param initial_value: The initial value of the variable.
        :type initial_value: number or a ndarray.

        :param name: Name of the variable.
        :type name: str.
        '''
        # Variable initial value.
        if np.shape(initial_value):
            self.initial_value = np.array(initial_value)
        else:
            self.initial_value = np.array([initial_value])

        # Output value of this operation in session execution.
        self.output_value = None

        # Nodes that receive this variable node as input.
        self.output_nodes = []

        # Variable name.
        self.name = name

        # Graph the variable belongs to.
        self.graph = DEFAULT_GRAPH

        # Add to the currently active default graph.
        self.graph.variables.append(self)
        if trainable:
            self.graph.trainable_variables.append(self)

    def compute_output(self):
        ''' Compute and return the variable value.
        '''
        if self.output_value is None:
            self.output_value = self.initial_value
        return self.output_value

# ------------------------------------------------------------------------------
# Placeholder node
# ------------------------------------------------------------------------------

class Placeholder(object):
    ''' Placeholder node in computational graph. It has to be provided a value when
        when computing the output of a graph.
    '''
    def __init__(self, name=None):
        ''' Placeholdef constructor.
        '''
        # Output value of this operation in session execution.
        self.output_value = None

        # Nodes that receive this placeholder node as input.
        self.output_nodes = []

        # Placeholder node name.
        self.name = name

        # Graph the placeholder node belongs to.
        self.graph = DEFAULT_GRAPH

        # Add to the currently active default graph.
        self.graph.placeholders.append(self)


# ------------------------------------------------------------------------------
# Function for gradients computation.
# ------------------------------------------------------------------------------

def compute_gradients(target_op):
    ''' Backpropagation implementation computing gradient of target operation wrt
        all the other connected nodes.

    :param target_op: The target operation whose gradient wrt other nodes would
                      be computed.
    :type target_op: Any operation type.

    :return grad_table: A table containing node objects and gradients.
    :type grad_table: dict.
    '''
    # A dict containing a mapping between node and gradient value of target_op wrt the node's output.
    # NOTE: It is the gradient wrt the node's OUTPUT NOT input.
    grad_table = {}

    # The gradient wrt target_op itself is 1.
    grad_table[target_op] = np.ones_like(target_op.output_value)

    # Perform a breadth-first search staring from the target_op in graph.
    # Queue for node traverasl.
    queue = Queue()
    queue.put(target_op)

    # Set for visited nodes.
    visited = set()
    visited.add(target_op)

    while not queue.empty():
        node = queue.get()

        # Compute gradient wrt the node's output.
        if node != target_op:
            grads_wrt_node_output = []
            for output_node in node.output_nodes:
                # Retrieve the gradient wrt output_node's OUTPUT.
                grad_wrt_output_node_output = grad_table[output_node]
                # Compute the gradient wrt current node's output.
                grad_wrt_node_output = output_node.compute_gradient(grad_wrt_output_node_output)
                input_node_index = output_node.input_nodes.index(node)
                grads_wrt_node_output.append(grad_wrt_node_output[input_node_index])

            # Sum all gradients wrt node's output.
            tot_grad_wrt_node_output = sum(grads_wrt_node_output)
            grad_table[node] = tot_grad_wrt_node_output

        # Put adjecent nodes to queue.
        if hasattr(node, 'input_nodes'):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table


