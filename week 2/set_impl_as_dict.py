# An alternate implementation of Sets.py
#
# Implements set operations using dictionary as the underlying data structure.
#

class Set:

    def __init__(self, collection=[]):
        self.dict = dict() # {}
        for elem in collection:
            #print "inserting ",elem
            self.dict[elem] = True

    # add an element to the set
    def add(self, elem):
        if elem not in self.dict:
            self.dict[elem] = True

    # remove an element from the set
    def pop(self):
        return self.dict.popitem()

    # remove specific element from the set
    def remove(self, elem):
        assert elem in self.dict, "The element must be in the Set!"
        del self.dict[elem]

    # return a set that contains both sets
    def union(self, other):
        unionSet = Set(self)

        for elem in other.dict:
            unionSet.add(elem)

        return unionSet

    # return a set that contains the intersection of the two sets
    def intersection(self, other):
        interSet = Set()

        for elem in self.dict:
            if elem in other.dict:
                interSet.add(elem)

        return interSet

    # return a set with the elements that are not in the other set
    def difference(self, other):
        diffSet = Set()
        union_set = self.union(other)
        
        for elm in self.dict:
            if elm not in union_set:
                diffSet.add(elm)

        return diffSet

    # return a set with items from both sets that do not appear in both sets
    def symmetric_difference(self, other):
        symdiffSet = Set()

        union_set = self.union(other)
        intersect_set = self.intersection(other)
        
        for elm in union_set:
            if elm not in intersect_set:
                symdiffSet.add(elm)

        return symdiffSet

    # clear the set
    def clear(self):
        self.dict = {}

    # check if the set is a subset of the other set
    # all items of the set are contained in the other set
    def issubset(self, other):
        if not isinstance(other, Set):  # Ensure other is a Set instance
            return TypeError("Argument must be a Set instance")

        for elem in self.dict:
            if elem not in other.dict:
                return False
        return True

    # check if the set is a superset of the other set
    # all items of the other set are contained in the set
    def issuperset(self, other):
        return other.issubset(self)

    # check is there are no common elements between the two sets
    def isdisjoint(self, other):
        if not isinstance(other, Set):  # Ensure other is a Set instance
            return TypeError("Argument must be a Set instance")

        for elem in self.dict:
            if elem in other.dict:
                return False
        return True

    # return a new set with the current elements
    def copy(self):
        return set(self)
    
    # how to print a set
    def __repr__(self):

        elements = sorted(self.dict.keys(),key=lambda elem:str(elem))
        return "Set(" + repr(elements) + ")"

    # iterator - return the next element
    def __iter__(self):
        return iter(self.dict)

    # define the contains check
    def __contains__(self, elem):
        return elem in self.dict

    # return the length of the set
    def __len__(self):
        return len(self.dict)

    # check if two sets are equal
    def __eq__(self, other):
        return self.dict == other.dict

#--------------------------------------------------------------------------

if __name__ == "__main__":
    exec(open("week 2\\set_tests.py").read())
    #from set_tests import *
    test1()
    test2()
    test3()
    test4()

    '''a = Set([1,2,3])
    b = Set([1,5,3])
    c = a.union(b)
    print(c)'''




