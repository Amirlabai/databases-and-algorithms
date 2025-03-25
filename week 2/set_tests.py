
# These are tests that should be passed by any Set implementation !
# Currently we have two Set implementations:
#      1. using list as the underlying data structure
#      2. using dict as the underlying data structure

def test1():
    A = Set([1, 2, 9, 1, 3, 1, 4, 1, 2, 9, 1, 3])
    B = Set([9, 5, 8, 9, 3, 5, 3, 1, 9, 5])
    print( "A =", A)
    print( "B =", B)
    print( "is 5 in A?", 5 in A)
    print( list(map(A.__contains__, [1,2,3,4,5,6,7])))
    print( list(map(A.__contains__, [10,11,12,13,14])))
    print( "Union A and B:", A.union(B))
    print( "Union B and A:", B.union(A))
    print( "Intersection A and B:", A.intersection(B))
    print( "Intersection B and A:", B.intersection(A))
    print( "Difference A and B:", A.difference(B))
    print( "Difference B and A:", B.difference(A))
    A.remove(1)
    print( "Removed 1 from A")
    print( "Intersection A and B:", A.intersection(B))
##    d = dict([(1,1), (2,2), (8,8)])
##    print( "Subtracting a dictionary from Set:")
##    print( "d =", d)
##    print( "Difference B and d:", B.difference(d))

def test2():
    A = Set('abracadabra')
    B = Set('hooplavoofla')
    print( "A =", A)
    print( "B =", B)
    print( "is 'z' in A?", 'z' in A)
    print( list(map(A.__contains__, list('abcdr'))))
    print( list(map(A.__contains__, list('123eyz'))))
    print( "Union A and B:", A.union(B))
    print( "Union B and A:", B.union(A))
    print( "Intersection A and B:", A.intersection(B))
    print( "Intersection B and A:", B.intersection(A))
    print( "Difference A and B:", A.difference(B))
    print( "Difference B and A:", B.difference(A))

def test3():  # This is the test from the lecture notes
    s1 = Set([])
    s1.add(17)
    s1.add(18)
    s1.add(18)              # adding 18 twice!
    assert 17 in s1
    assert len(s1) == 2
    B = [2, 4, 6, 8, 2, 6]  # list container
    C = [4, 8, 2, 6]        # list container
    s2 = Set(B)
    s3 = Set(C)
    assert s2 == s3
    s3.add(100)
    #assert s2.issubset(s3) # issubset is not implemented yet (left as an exercise!)
    s3.remove(100)
    assert s2 == s3
    print( "Test 3 PASSED")

def test4():  # This test is for the methods that were left as home work
    A = Set([1, 2, 9, 1, 3, 1, 4, 1, 2, 9, 1, 3])
    B = Set([9, 5, 8, 9, 3, 5, 3, 1, 9, 5])
    C = Set([9, 1, 3, 1, 4, 1, 9, 1, 3])
    D = Set(['a', 'b', 'x', 'y'])
    print("A =", A)
    print("B =", B)
    print("C =", C)
    print("D =", D)
    assert C.issubset(A), "issubset not working"
    assert A.issuperset(C), "not suiper"
    assert A.isdisjoint(D),  " not disjoint"
    print("A union B =", A.union(B))
    print("A intersection B =", A.intersection(B))
    print("A intersection D =", A.intersection(D))
    print("A symmetric_difference B =", A.symmetric_difference(B))
    E = D.copy()
    E.add('hello')
    print("D =", D)
    print("E =", E)
