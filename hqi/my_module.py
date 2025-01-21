# -*- coding : utf-8 -*-

"""
Modules have doctrings!
"""


def some_function(some_integer):
    """
    Functions have docstrings.

    Variable and functions are named in snake case
    
    Args:
        some_integer (int): an integer

    Returns:
        int: twice that integer.
    """
    return 2 * some_integer


class MyClass:
    """
    Classes have docstrings.
    Classes are named in camel case.
    """

    def __init__(self, *args):
        pass
