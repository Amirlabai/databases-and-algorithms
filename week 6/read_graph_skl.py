#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      ofertzur
#
# Created:     28/05/2018
# Copyright:   (c) ofertzur 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from graphics import *
from point import Point
from line import Line
from graph import *
import tkinter as tk
import sys
import re
sys.path.append(r'C:\Users\amirl\\OneDrive\\Documents\\GitHub\\databases-and-algorithms\week_5')
from plain_tree import PlainTree

kwargs = {}

def parse_ints(s):
    t = re.sub('[(),]', ' ', s)
    ints = [int(x) for x in t.split()]
    return ints

def read_graph(graph_name,directed=False):
    g = Graph(directed)
    verts = dict()

    try:
        with open(graph_name, 'r') as f:
            lines = f.readlines()

        # Parse vertices from the first line
        vertex_coords = parse_ints(lines[0])
        for i in range(0, len(vertex_coords), 2):
            x, y = vertex_coords[i], vertex_coords[i + 1]
            point = Point(x, y)
            verts[(x, y)] = g.insert_vertex(point)  # Add the Point as a vertex to the graph

        # Parse edges from the second line
        edge_indices = parse_ints(lines[1])
        #vertices_list = list(verts.values())  # Get the list of vertices in insertion order
        for i in range(0, len(edge_indices), 4):
            v1_idx, v2_idx = (edge_indices[i], edge_indices[i + 1]),(edge_indices[i+2], edge_indices[i + 3])
            v1 = verts[v1_idx]
            v2 = verts[v2_idx]
            edge = Line(v1.element(), v2.element())  # Create a Line using the Points
            g.insert_edge(v1, v2, edge)  # Add the Line as an edge to the graph

        # Draw the graph
        for vertex in g.vertices():
            pt = vertex.element()
            pt.draw()  # Draw each Point
            pt.text(f"({pt.x},{pt.y})")  # Show coordinates as text

        for edge in g.edges():
            edge.element().draw()  # Draw each Line

    except FileNotFoundError:
        print(f"Error: File '{graph_name}' not found.")
    except IndexError:
        print("Error: Malformed graph file. Check vertex and edge definitions.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return g, verts

def BFS(graph, start, discovered):
    """
    Perform a breadth-first search (BFS) on the graph starting from the given vertex.

    Parameters:
    - graph: The graph object.
    - start: The starting vertex for the BFS.
    - discovered: A dictionary mapping each discovered vertex to the edge that was used to discover it.

    Returns:
    - None (discovered is updated in place).
    """
    queue = [start]  # Initialize the queue with the starting vertex
    while queue:
        current_vertex = queue.pop(0)  # Dequeue a vertex
        for edge in graph.incident_edges(current_vertex):  # Iterate over all edges connected to the current vertex
            opposite_vertex = edge.opposite(current_vertex)  # Get the vertex on the other side of the edge
            if opposite_vertex not in discovered:  # If the vertex has not been discovered yet
                edge.element().draw(width=3, fill='light green')  # Draw the edge with a growing pixel width
                discovered[opposite_vertex] = edge  # Mark the edge that discovered this vertex
                queue.append(opposite_vertex)  # Enqueue the newly discovered vertex
    return discovered  # Return the discovered dictionary

def DFS(graph, start, discovered,i=5,**kwargs):
    """
    Perform a depth-first search (DFS) on the graph starting from the given vertex.

    Parameters:
    - graph: The graph object.
    - start: The starting vertex for the DFS.
    - discovered: A dictionary mapping each discovered vertex to the edge that was used to discover it.

    Returns:
    - None (discovered is updated in place).
    """
    for edge in (graph.incident_edges(start)):  # Iterate over all edges connected to the start vertex
        opposite_vertex = edge.opposite(start)  # Get the vertex on the other side of the edge
        if opposite_vertex not in discovered:  # If the vertex has not been discovered yet
            edge.element().draw(**kwargs)  # Draw the edge with a growing pixel width
            discovered[opposite_vertex] = edge  # Mark the edge that discovered this vertex
            DFS(graph, opposite_vertex, discovered,i*0.9)  # Recursively explore the vertex

def color_line(graph, start, end):
    """
    Color the line between two points in the graph.

    Parameters:
    - graph: The graph object.
    - start: The starting point (Point object).
    - end: The ending point (Point object).

    Returns:
    - None
    """
    try:
        Vst = graph.get_edge(start,end)
        Vst.element().draw(fill='red')  # Create a Line object between the two points
        #line.draw(width=grow_pixel(5), fill=get_next_color(0))  # Draw the line with a specific width and color
    except Exception as e:
        print(f"Error: {e}")
        print("The line between the two points could not be colored. Check if the points are connected.")

def graph_to_tree(graph, vertex=None,discovered=None):
    if gtree is None:
        gtree = PlainTree()
        position = gtree.add_root([graph, 0])
    
    try:
        for vertex in graph.vertices():
            if vertex.element() not in gtree.get_children(position):
                new_position = gtree.add_child(position, [vertex.element(), 0])  # Initially size 0 for dirs
                graph_to_tree(vertex.element(), gtree, new_position)
                position.get_element()[1] += new_position.get_element()[1]  # Update directory size after recursion
    except Exception as e:
        print(f"Error while converting graph to tree: {e}")
        print("Ensure the graph structure is valid and connected.")

def main():
    g, verts = read_graph("week 6/data/graph12.dat",True)
    v = verts[298,216]

    def clear_and_redraw_base():
        canvas.delete("all")
        # Draw all edges in black
        for edge in g.edges():
            edge.element().draw()
        # Draw all vertices as white dots
        for vertex in g.vertices():
            pt = vertex.element()
            kwargs.pop('arrow', 1)
            kwargs.pop('arrowshape',1)
            pt.draw()
            pt.text(f"({pt.x},{pt.y})")
        # Draw starting vertex as a red box
        pt = v.element()
        size = 8
        canvas.create_rectangle(pt.x - size, pt.y - size, pt.x + size, pt.y + size, fill='red', outline='red')

    def show_dfs(**kwargs):
        discovered = dict()
        discovered[v] = None
        clear_and_redraw_base()
        DFS(g, v, discovered,**kwargs)
        show_canvas()

    def show_bfs(**kwargs):
        discovered = dict()
        discovered[v] = None
        clear_and_redraw_base()
        BFS(g, v, discovered)
        show_canvas()
    
    def event_show_dfs(event):
        show_dfs(**kwargs)
    
    def event_show_bfs(event):
        show_bfs(**kwargs)

    def clsoe_window():
        rootWindow.destroy()
    
    if g.is_directed():
        kwargs.setdefault('arrow', 'last')
        kwargs.setdefault('arrowshape', [6,15,5])
    kwargs.setdefault('fill', 'light green')
    # Use the rootWindow from graphics.py for buttons
    button1 = tk.Button(rootWindow, text="DFS", command=lambda: show_dfs(**kwargs))
    button1.pack(padx=5, pady=5,side=tk.RIGHT)
    button2 = tk.Button(rootWindow, text="BFS", command=show_bfs)
    button2.pack(padx=5, pady=5,side=tk.RIGHT)
    button3 = tk.Button(rootWindow, text="Close", command=clsoe_window, bg='gray')
    button3.pack(padx=5, pady=5,side=tk.LEFT)

    rootWindow.bind("<Right>", event_show_dfs)  # Bind the Enter key to show DFS
    rootWindow.bind("<Left>", event_show_bfs)  # Bind the Enter key to show BFS
    rootWindow.bind("<Escape>", lambda e: rootWindow.destroy())  # Bind the Escape key to close the window
    show_canvas()

def main2():
    g, verts = read_graph("week 6/data/graph12.dat")
    v = verts[278,454]
    #FilesTree(g, v)


if __name__ == '__main__':
    main()