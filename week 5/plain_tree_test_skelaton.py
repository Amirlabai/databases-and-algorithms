from plain_tree import PlainTree
import os

IMPORT_PATH = ""


def test1():
    t = PlainTree()
    p0 = t.add_root(0)
    p1 = t.add_child(p0, 100)
    p2 = t.add_child(p0, 200)
    p3 = t.add_child(p0, 300)
    t.add_child(p2, 400)
    t.add_child(p2, 500)
    t.add_child(p3, 600)
    t.add_child(p3, 700)
    print(t)

def test2():
    t = PlainTree()
    pos = dict()
    pos[0] = t.add_root(0)
    for n in range(200):
        if 3*n+3 > 199:
            break
        pos[3*n+1] = t.add_child(pos[n], 3*n+1)
        pos[3*n+2] = t.add_child(pos[n], 3*n+2)
        pos[3*n+3] = t.add_child(pos[n], 3*n+3)

    print(t)



# use the following two function in order to calculate
# the sizes of the directories (and update the tree)
#
# use the Print_Sizes function to use the sum_tree and print the
# result
def sum_tree(ftree, position):
    node = ftree._validate(position)
    total_size = 0

    if not os.path.isdir(node.element[0]):  # If it's a file
        total_size = node.element[1]  # File size is already stored
    else:
        for child_position in ftree.get_children(position):
            total_size += sum_tree(ftree, child_position)  # Recursive call
        node.element[1] = total_size  # Update directory size in the tree

    return total_size

def Print_Sizes(ftree, position=None, indent=""):
    """Print the files and directories with their sizes."""

    if position is None:
        position = ftree.get_root()

    node = ftree._validate(position)
    name = node.element[0]
    size = node.element[1]

    print(f"{indent}{name} - {size} bytes")

    if os.path.isdir(name):  # Only print children if it's a directory
        for child_position in ftree.get_children(position):
            Print_Sizes(ftree, child_position, indent + "  ")  # Add indentation


# generate a tree of the directory "dir"
def FilesTree(dir, ftree=None, position=None):
    if ftree is None:
        ftree = PlainTree()
        position = ftree.add_root([dir, 0])  # Initialize with dir name and size 0

    try:
        for file in os.listdir(dir):
            path = os.path.join(dir, file)
            size = os.path.getsize(path)
            if os.path.isdir(path):
                # Add directory node and recursively explore
                new_position = ftree.add_child(position, [file,0])  # Initially size 0 for dirs
                FilesTree(path, ftree, new_position)
                position.get_element()[1] += new_position.get_element()[1]  # Update directory size after recursion
                #tot_size = 0  # Reset total size for the next directory
            else:
                # Add file node with its size
                ftree.add_child(position, [file, size])
                position.get_element()[1] += size

        sum_tree(ftree,position)
    except PermissionError:
        print(f"PermissionError: Cannot access {dir}")
    except FileNotFoundError:
        print(f"FileNotFoundError: {dir} not found")

    return ftree  # Return the populated tree


if __name__ == "__main__":

    from config import IMPORT_PATH
    #test1()
    #test2()
    #ftree = FilesTree("c:/Windows")
    #print ftree
    ftree= FilesTree(IMPORT_PATH) #C:\Users\amirl\Documents\Education\סמסטר ו'\0.אספקה\הרצאות
    print(ftree)
    #sum_tree(ftree,ftree.get_root())
    #Print_Sizes(ftree)


