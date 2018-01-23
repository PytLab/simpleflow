#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .graph import *
from .operations import *
from .session import *
from .train import *

# Create a default graph.
import builtins
DEFAULT_GRAPH = builtins.DEFAULT_GRAPH = Graph()

