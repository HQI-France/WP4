# -*- coding : utf-8 -*-

"""
Test for this module
"""


from hqi.my_module import some_function, MyClass


class TestMyClass:
    """
    Testing MyClass
    """

    def test_instantiation(self):
        """
        Testing the constructor
        """
        MyClass(1, 1, 2)
        MyClass()


def test_some_function():
    """
    Testing some_function
    """
    assert 4 == some_function(2)
