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
    def __init__(self, collection=[]):
        self.elements = []
        for elem in collection:
            if elem not in self.elements:
                self.elements.append(elem)

    # add an element to the set
    def add(self, elem):
        if elem not in self.elements:
                self.elements.append(elem)

    # remove an element from the set
    def pop(self):
        if not self.elements:
            return None, self.elements #empty list
        try:
            popped_element = self.elements.pop(-1)
            return popped_element, self.elements
        except IndexError:
            return None, self.elements #index out of range

    # remove specific element from the set
    def remove(self, elem):
        assert elem in self, "The element must be in the Set!"
        if not self.elements:  # Check if the list is empty
            return self.elements
        else:
            return self.elements[:-1]

    # return a set that contains both sets
    def union(self, other):
        if isinstance(other, Set):
            unionSet = Set(self)
            for elem in other.elements:
                unionSet.add(elem)
            return unionSet
        return None

    # return a set that contains the intersection of the two sets
    def intersection(self, other):
        interSet = Set()

        for elem in self.elements:
            if elem in other.elements:
                interSet.add(elem)

        return interSet


    # return a set with the elements that are not in the other set
    def difference(self, other):
        diffSet = Set()
        union_set = self.union(other)
        
        for elm in self.elements:
            if elm not in union_set:
                diffSet.add(elm)

        return diffSet


    # clear the set
    def clear(self):
        self.elements = []

    # check if the set is a subset of the other set
    def issubset(self, other):
        if not isinstance(other, Set):  # Ensure other is a Set instance
            return TypeError("Argument must be a Set instance")

        for elem in self.elements:
            if elem not in other.elements:
                return False
        return True

    # check if the set is a superset of the other set
    def issuperset(self, other):
        return other.issubset(self)

    # check is there are no common elements between the two sets
    def isdisjoint(self, other):
        if not isinstance(other, Set):  # Ensure other is a Set instance
            return TypeError("Argument must be a Set instance")

        for elem in self.elements:
            if elem in other.elements:
                return False
        return True

    # return a set with items from both sets that do not appear in both sets
    def symmetric_difference(self, other):
        symdiffSet = Set()

        union_set = self.union(other)
        intersect_set = self.intersection(other)
        
        for elm in union_set:
            if elm not in intersect_set:
                symdiffSet.add(elm)

        return symdiffSet

    # return a new set with the current elements
    def copy(self):
        return Set(self)


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
    exec(open("week 2\\set_tests.py").read())
    #from set_tests import *
    test1()
    test2()
    #test3()

