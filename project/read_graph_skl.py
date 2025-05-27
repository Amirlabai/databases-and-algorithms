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
from plain_tree import PlainTree

import pandas as pd


def parse_ints(s):
    t = re.sub('[(),]', ' ', s)
    ints = [int(x) for x in t.split()]
    return ints

def read_table(graph_name):
    g = Graph(True)
    verts = dict()
    running_days = 0
    predecessor = '0'  # Initialize predecessor to None
    try:
        df = pd.read_csv(graph_name, encoding='utf-8')  # Read the CSV file into a DataFrame
        #df = df.values  # Convert DataFrame to numpy array, ignoring headers

        for i,row in enumerate(df.iterrows()):
            leangth = int(row[1]['Duration'].split(' ')[0])
            if predecessor != row[1]['Predecessors']:
                predecessor = row[1]['Predecessors']
                running_days += leangth
            print(f"Row {i}: {row[1]['Tid']} - Duration: {leangth}")
            print(f"predeccsor for {row[1]['Tid']}: {row[1]['Predecessors']} - running time: {running_days}")
            y = row[1]['Tid']
            x = (i+1)
            if i == 0:
                point = Point(30, y+30)  # Skip the header row
            else:
                point = Point(running_days*10+30, y*17+30)
            verts[str(y)] = g.insert_vertex(point)  # Add the Point as a vertex to the graph

        for i,row in enumerate(df.iterrows()):
            if i == 0:
                continue
            else:
                for j in row[1]['Predecessors'].split(','):
                    v1 = verts[j] 
                    v2 = verts[str(row[1][0])]
 
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
            edge.element().draw(width=3, fill='light green')  # Draw the edge with a growing pixel width
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
        print("Ensure the graph structure is valid and all vertices are connected.")

def main():
    g, verts = read_table(r"project\tasks.csv")
    v = verts['1']

    def clear_and_redraw_base():
        canvas.delete("all")
        # Draw all edges in black
        for edge in g.edges():
            edge.element().draw(width=3, fill='blue')
        # Draw all vertices as white dots
        for vertex in g.vertices():
            pt = vertex.element()
            pt.draw(fill='black')
            pt.text(f"({pt.x},{pt.y})")
        # Draw starting vertex as a red box
        pt = v.element()
        size = 8
        canvas.create_rectangle(pt.x - size, pt.y - size, pt.x + size, pt.y + size, fill='red', outline='red')

    def show_dfs():
        discovered = dict()
        discovered[v] = None
        clear_and_redraw_base()
        DFS(g, v, discovered)
        show_canvas()

    def show_bfs():
        discovered = dict()
        discovered[v] = None
        clear_and_redraw_base()
        BFS(g, v, discovered)
        show_canvas()
    
    def event_show_dfs(event):
        show_dfs()
    
    def event_show_bfs(event):
        show_bfs()

    def clsoe_window():
        rootWindow.destroy()

    # Use the rootWindow from graphics.py for buttons
    button1 = tk.Button(rootWindow, text="DFS", command=show_dfs)
    button1.pack(padx=5, pady=5,side=tk.RIGHT)
    button2 = tk.Button(rootWindow, text="BFS", command=show_bfs)
    button2.pack(padx=5, pady=5,side=tk.RIGHT)
    button3 = tk.Button(rootWindow, text="Close", command=clsoe_window, bg='gray')
    button3.pack(padx=5, pady=5,side=tk.LEFT)

    rootWindow.bind("<Right>", event_show_dfs)  # Bind the Enter key to show DFS
    rootWindow.bind("<Left>", event_show_bfs)  # Bind the Enter key to show BFS
    rootWindow.bind("<Escape>", lambda e: rootWindow.destroy())  # Bind the Escape key to close the window
    show_canvas()


if __name__ == '__main__':
    main()