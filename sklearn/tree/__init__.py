"""
The :mod:`sklearn.tree` module includes decision tree-based models for
classification and regression.
"""

from .tree import DecisionTreeClassifier
from .tree import DecisionTreeRegressor
from .tree import ExtraTreeClassifier
from .tree import ExtraTreeRegressor
from .lin_reg_tree import LinearDecisionTreeRegressor
from .export import export_graphviz

__all__ = ["DecisionTreeClassifier", "DecisionTreeRegressor",
           "ExtraTreeClassifier", "ExtraTreeRegressor", "LinearDecisionTreeRegressor", "export_graphviz"]
