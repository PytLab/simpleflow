#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Session to execute a computational graph.
'''
from functools import reduce

from .operations import Operation, Variable, Placeholder

class Session(object):
    ''' A session to compute a particular graph.
    '''
    def __init__(self):
        ''' Session constructor.
        '''
        # Graph the session computes for.
        self.graph = DEFAULT_GRAPH

    def __enter__(self):
        ''' Context management protocal method called before `with-block`.
        '''
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        ''' Context management protocal method called after `with-block`.
        '''
        self.close()

    def close(self):
        ''' Free all output values in nodes.
        '''
        all_nodes = (self.graph.constants + self.graph.variables +
                     self.graph.placeholders + self.graph.operations +
                     self.graph.trainable_variables)
        for node in all_nodes:
            node.output_value = None

    def run(self, operation, feed_dict=None):
        ''' Compute the output of an operation.

        :param operation: A specific operation to be computed.
        :type operation: object of `Operation`, `Variable` or `Placeholder`.

        :param feed_dict: A mapping between placeholder and its actual value for the session.
        :type feed_dict: dict.
        '''
        # Get all prerequisite nodes using postorder traversal.
        postorder_nodes = _get_prerequisite(operation)

        for node in postorder_nodes:
            if type(node) is Placeholder:
                node.output_value = feed_dict[node]
            else:  # Operation and variable
                node.compute_output()

        return operation.output_value

def _get_prerequisite(operation):
    ''' Perform a post-order traversal to get a list of nodes to be computed in order.
    '''
    postorder_nodes = []

    # Collection nodes recursively.
    def postorder_traverse(operation):
        if isinstance(operation, Operation):
            for input_node in operation.input_nodes:
                postorder_traverse(input_node)
        postorder_nodes.append(operation)

    postorder_traverse(operation)

    return postorder_nodes

