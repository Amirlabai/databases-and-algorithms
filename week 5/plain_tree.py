# Copyright 2013, Michael H. Goldwasser
#
# Developed for use with the book:
#
#    Data Structures and Algorithms in Python
#    Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser
#    John Wiley & Sons, 2013
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# samyz: Some modifications by Samy Zafrany (Dec 15, 2013)
# samyz: move nested classes outside

import tree
#from tree import Tree,Position
#from tree import *

#--------- Node class ------------

class Node:
    "Class for storing a tree node"

    def __init__(self, element, parent=None, children=None):
        self.element = element
        self.parent = parent
        self.children = children if children is not None else [] # list of Node objects

#--------- Position class ----------
class Position(tree.Position):
    """An abstraction representing the location of a single element."""

    def __init__(self, owner, node):
        """Constructor should not be invoked by user."""
        self.owner = owner
        self.node = node

    def get_element(self):
        """Return the element stored at this Position."""
        return self.node.element

    def __eq__(self, other):
        """Return True if other is a Position representing the same location."""
        return type(other) is type(self) and other.node is self.node


#--------- PlainTree class ----------

class PlainTree(tree.Tree):
    """Simple implementation of a tree structure."""

    #-------------------------- tree constructor --------------------------
    def __init__(self):
        """Create an initially empty tree."""
        self.root = None
        self.size = 0

    #-------------------------- public accessors --------------------------
    def __len__(self):
        """Return the total number of elements in the tree."""
        return self.size

    def get_root(self):
        """Return the root Position of the tree (or None if tree is empty)."""
        return self._make_position(self.root)

    def get_parent(self, p):
        """Return the Position of p's parent (or None if p is root)."""
        node = self._validate(p)
        return self._make_position(node.parent)

    def get_children(self, p):
        """Return the Position of p's children"""
        node = self._validate(p)
        for c in node.children:
            yield self._make_position(c)

    def num_children(self, p):
        """Return the number of children of Position p."""
        node = self._validate(p)
        return len(node.children)

    def add_root(self, e):
        """
        Place element e at the root of an empty tree and return new Position.
        Raise ValueError if tree nonempty.
        """
        if self.root is not None:
            raise ValueError('Root exists')
        self.size = 1
        self.root = Node(e) #,None,[])
        return self._make_position(self.root)

    def add_child(self, p, e):
        """
        add a new element e at the end of children of p
        Return the Position of new node.
        Raise ValueError if Position p is invalid
        """
        parent_node = self._validate(p)
        child_node = Node(e, parent_node, [])
        parent_node.children.append(child_node)
        #child_position =
        self.size += 1
        return self._make_position(child_node)

    def insert_child(self, p, e, i=0):
        """
        Insert a new element e at position i in the children list of of p
        Return the Position of new node.
        Raise ValueError if Position p is invalid
        """
        parent_node = self._validate(p)
        child_node = Node(e, parent_node, [])
        parent_node.children.insert(i, child_node)
        child_position = self._make_position(child_node)
        self.size += 1
        return child_position

    def replace(self, p, e):
        "Replace the element at position p with e, and return old element."
        node = self._validate(p)
        old = node.element
        node.element = e
        return old

    def delete(self, p):
        """
        Delete the node at Position p, and replace it with its child, if any.
        Return the element that had been stored at Position p.
        Raise ValueError if Position p is invalid or p has two children.
        """
        node = self._validate(p)
        if node is self.root:
            raise ValueError('Cannot delete root!')
        # finding the location of the deleted node in the parent's children list
        num_children = len(node.parent.children)
        i = 0
        while(i < num_children):
            if node.parent.children[i] == node:
                clist = node.parent.children[:i] + node.children
                if i < num_children:
                    clist += node.parent.children[i + 1:]
                node.parent.children = clist
                break
            else:
                i+=1

        for child in node.children:
            child.parent = node.parent   # child's grandparent becomes parent
        self.size -= 1
        node.parent = node  # convention for deprecated node (that's a delete)
        return node.element

    def __str__(self):
        lines = []
        for p in self.preorder():
            e = p.get_element()
            d = self.depth(p)
            if isinstance(e,list):
                e = "\t".join([str(xx) for xx in e])
            line = d*"   |" + "_" +  str(e)
            lines.append(line)
        return '\n'.join(lines)

    #------------------------------- utility methods -------------------------------
    def _validate(self, p):
        """Return associated node, if position is valid."""
        if not isinstance(p, Position):
            raise TypeError('p must be proper Position type')
        if p.owner is not self:
            raise ValueError('p does not belong to this tree (owner tree)')
        if p.node.parent is p.node:      # convention for deprecated nodes
            raise ValueError('p is no longer valid')
        return p.node

    def _make_position(self, node):
        """Return Position instance for given node (or None if no node)."""
        return Position(self, node) if node is not None else None