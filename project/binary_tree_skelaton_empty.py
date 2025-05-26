#-------------------------------------------------------------------------------
# Name:        module2
# Purpose:
#
# Author:      ofer
#
# Created:     16/12/2013
# Copyright:   (c) ofer 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
class Node():
    def __init__(self, val):
        self.value = val
        self.left = None
        self.right = None
        self.parent = None

    # return the left child
    def Left(self):
        return self.left
    # return the right child
    def Right(self):
        return self.right

    # set node to be the left child of current node
    def set_left(self,node):
        if self.left is None:
            self.left = node
            self.left.parent = self
        else:
            print("Error: Left child already exists for node with value:", self.value)

    # set node to be the right child of current node
    def set_right(self,node):
        if self.right is None:
            self.right = node
            if node is not None:
                node.parent = self
        else:
            print("Error: Right child already exists for node with value:", self.value)

    # a node is visited after the left child and before thr right child
    def inorder(self):
        result = []
        # Traverse left subtree
        if self.left:
            result.extend(self.left.inorder())
        # Visit root node
        result.append(self.value)
        # Traverse right subtree
        if self.right:
            result.extend(self.right.inorder())
        return result

    # a node is visited before its children
    # afterwards, the left child and then the right child
    def PreOrder(self):
        result = []
        # Visit root node
        result.append(self.value)
        # Traverse left subtree
        if self.left:
            result.extend(self.left.PreOrder())
        # Traverse right subtree
        if self.right:
            result.extend(self.right.PreOrder())
        return result

    # the left child first
    #then the right child and the father at the end
    def PostOrder(self):
        result = []
        # Traverse left subtree
        if self.left:
            result.extend(self.left.PostOrder())
        # Traverse right subtree
        if self.right:
            result.extend(self.right.PostOrder())
        # Visit root node
        result.append(self.value)
        return result


tree = Node(4)
left = Node(3)
left1 = Node(1)
left2 = Node(8)
right2 = Node(9)
left1.set_left(left2)
left1.set_right(right2)
left.set_left(left1)
left.set_right(Node(20))
right = Node(7)
right.set_left(Node(6))
right.set_right(Node(30))
tree.set_left(left)
tree.set_right(right)


#               4
#       3              7
#     1   20         6   30
#    8 9

# should return: [8,1,9,3,20,4,6,7,30]
answer = tree.inorder()
print(answer)
assert answer== [8,1,9,3,20,4,6,7,30]

answer = tree.PreOrder()
print(answer)
assert answer == [4,3,1,8,9,20,7,6,30]

answer = tree.PostOrder()
print(answer)
assert answer == [8,9,1,20,3,6,30,7,4]

print("Success")
