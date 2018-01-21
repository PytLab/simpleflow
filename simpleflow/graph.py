#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Computational graph definition.
'''

class Graph(object):
    ''' Graph containing all computing nodes.
    '''
    def __init__(self):
        ''' Graph constructor.
        '''
        self.operations, self.constants, self.placeholders = [], [], []
        self.variables, self.trainable_variables = [], []

    def __enter__(self):
        ''' Reset default graph.
        '''
        global DEFAULT_GRAPH
        self.old_graph = DEFAULT_GRAPH
        DEFAULT_GRAPH = self
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        ''' Recover default graph.
        '''
        global DEFAULT_GRAPH
        DEFAULT_GRAPH = self.old_graph

    def as_default(self):
        ''' Set this graph as global default graph.
        '''
        return self

