# Based on open source code:
#
#     http://code.activestate.com/recipes/230113-implementation-of-sets-using-sorted-lists
#
# (With some modifications and simplifications from Rance D. Necaise Book)
#
# An alternate implementation of Sets.py
#
# Implements set operations using list as the underlying data structure.
#
# IMPORTANT NOTE:
#     Several methods and the init method accept a collection argument.
#     This collection object is assumed to be any Python collection, and not
#     necessarily a Set! Therefore we cannot apply Set methods on such objects, and
#     must be careful not to make any special assumptions about them.
#

class Set:

    # constructor
    def __init__(self, collection):
        pass # for student to fill

    # add an element to the set
    def add(self, elem):
        pass # for student to fill

    # remove an element from the set
    def pop(self):
        pass # for student to fill

    # remove specific element from the set
    def remove(self, elem):
        assert elem in self, "The element must be in the Set!"
        pass

    # return a set that contains both sets
    def union(self, other):
        pass # for student to fill


    # return a set that contains the intersection of the two sets
    def intersection(self, other):
        pass # for student to fill


    # return a set with the elements that are not in the other set
    def difference(self, other):
        pass # for student to fill


    # clear the set
    def clear(self):
        self.elements = []

    # check if the set is a subset of the other set
    def issubset(self, other):
        pass # for student to fill

    # check if the set is a superset of the other set
    def issuperset(self, other):
        pass # for student to fill

    # check is there are no common elements between the two sets
    def isdisjoint(self, other):
        pass # for student to fill

    # return a set with items from both sets that do not appear in both sets
    def symmetric_difference(self, other):
        pass # for student to fill

    # return a new set with the current elements
    def copy(self):
        pass # for student to fill


    def __repr__(self):
        elems = sorted(self.elements, key=lambda elem: str(elem))
        return "Set(" + repr(elems) + ")"

    def __iter__(self):
        return iter(self.elements)

    def __contains__(self, elem):
        return elem in self.elements

    def __len__(self):
        return len(self.elements)

    def __eq__(self, other):
        return sorted(self.elements) == sorted(other.elements)

#--------------------------------------------------------------------------

if __name__ == "__main__":
    exec(open("./set_tests.py").read())
    #from set_tests import *
    test1()
    test2()
    #test3()

