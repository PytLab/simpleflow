#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .graph import *
from .operations import *
from .session import *

# Create a default graph.
import builtins
builtins.DEFAULT_GRAPH = Graph()

