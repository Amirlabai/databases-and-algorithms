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

import re

def parse_ints(s):
    t = re.sub('[(),]', ' ', s)
    ints = [int(x) for x in t.split()]
    return ints

def read_graph(graph_name):
    g = Graph()
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
            vertex.element().draw()  # Draw each Point
        for edge in g.edges():
            edge.element().draw()  # Draw each Line

    except FileNotFoundError:
        print(f"Error: File '{graph_name}' not found.")
    except IndexError:
        print("Error: Malformed graph file. Check vertex and edge definitions.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return g, verts

def DFS(graph, start, discovered,i=5):
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
            edge.element().draw(width=grow_pixel(i), fill=get_next_color(int(i*2)))  # Draw the edge with a growing pixel width
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

def main():
    g,verts = read_graph("week 6/data/graph1.dat") #week 6\data\graph1.dat

    discovered = dict()
    v = verts[278,454]
    discovered[v] = None
    DFS(g,v,discovered)

    color_line(g,verts[520,226],verts[346,145])

    #draw graph
    show_canvas()

    # show the result using the following:
    #mainloop()


if __name__ == '__main__':
    main()