# This is a program for generating random plannar graphs
# Written by samyz
# Based on  graphs.py from the book:
#    Data Structures and Algorithms in Python
#    Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser
#    John Wiley & Sons, 2013

from graph import Graph
from graphics import *
from point import Point
from line import Line
from math import sin, cos, radians, sqrt
import random
import re
from queue import PriorityQueue
import AdaptableHeapPriorityQueue as AQ

def distance(a,b):
    return sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

def is_singleton(g, v):
    if not g.incident_edges(v,True) and not g.incident_edges(v,False):
        return True
    return False

def add_random_cycle(g, k):
    V = g.vertices()
    n = len(V)
    C = []
    for j in range(k):
        i = random.randint(0,n-j-1)
        C.append(V.pop(i))
    print(len(C))
    for j in range(k-1):
        e = g.get_edge(C[j], C[j+1])
        if e is None:
            e = g.insert_edge(C[j], C[j+1])
        draw_edge(e, fill="green", arrow="last", width=1)
    e = g.get_edge(C[k-1], C[0])
    if e is None:
        e = g.insert_edge(C[k-1], C[0])
    draw_edge(e, fill="green", arrow="last", width=1)


def random_graph(file, **kwargs):
    g = Graph()

    kwargs.setdefault('num_verts', 70)
    kwargs.setdefault('min_dist', 50)
    kwargs.setdefault('max_dist', 300)
    kwargs.setdefault('num_edges', 1000)
    kwargs.setdefault('probability', 0.8)

    num_verts = kwargs['num_verts']
    num_edges = kwargs['num_edges']
    min_dist = kwargs['min_dist']
    max_dist = kwargs['max_dist']
    probability = kwargs['probability']
    ctr = 0
    P = []
    while True:
        x, y = random.randint(60, 740), random.randint(60, 540)
        p = Point(x,y)
        skip = False
        for q in P:
            if distance(p,q)<min_dist:
                skip = True
                break
        if skip: continue
        P.append(p)
        p.draw()
        p.text("P(%d,%d)" % (x,y), font="Consolas 8 bold")
        g.insert_vertex(p)
        ctr += 1
        if ctr>num_verts:
            break

    V = g.vertices()
    edge_ctr = 0
    for i in range(len(V)):
        a = V[i]
        for j in range(i+1, len(V)):
            b = V[j]
            l = Line(a.element(), b.element())
            d = l.length()
            if random.uniform(0,1) < probability and d<max_dist:
                l.draw(fill="red", width=1)
                g.insert_edge(a, b, l.length())
                edge_ctr += 1
                if edge_ctr > num_edges:
                    print("Reached max num_edges:", num_edges)
                    break

    #add_random_cycle(g, 3)
    #add_random_cycle(g, 4)
    #add_random_cycle(g, 5)

    f = open(file, 'w')
    points_text = " ".join(["(%d,%d)" % (p.x,p.y) for p in P])
    f.write(points_text + '\n')
    E = []
    for e in g.edges():
        a,b = e.endpoints()
        p,q = a.element(), b.element()
        E.append("((%d,%d),(%d,%d))" % (p.x, p.y, q.x, q.y))

    edges_line = " ".join(E) + '\n'
    f.write(edges_line)
    f.close()

    canvas.tag_raise('POINT')
    canvas.tag_raise('TEXT')
    show_canvas()

# Parse the list of integers inside a string like '(10,20)' or '((10,20),(30,40))'
def parse_ints(s):
    t = re.sub('[(),]', ' ', s)
    ints = [int(x) for x in t.split()]
    return ints

def read_graph(file, directed=False):
    g = Graph(directed)
    f = open(file)
    points = f.readline().split()
    lines = f.readline().split()
    f.close()
    vert = dict()

    for point in points:
        x,y = parse_ints(point)
        p = Point(x,y)
        vert[x,y] = g.insert_vertex(p)

    for l in lines:
        x1,y1,x2,y2 = parse_ints(l)
        v1 = vert[x1,y1]
        v2 = vert[x2,y2]
        dist = distance(v1.element(), v2.element())
        g.insert_edge(v1, v2, dist)

    return g

def draw_graph(g):
    for v in g.vertices():
        p = draw_vertex(v)
        p.text("P(%d,%d)" % (p.x, p.y), font="Consolas 8 bold")
    for e in g.edges():
        if g.is_directed():
            draw_edge(e, fill="red", width=1, arrow="last")
        else:
            draw_edge(e, fill="red", width=1)

    canvas.tag_raise('POINT')
    canvas.tag_raise('TEXT')

def draw_vertex(v, **kwargs):
    p = v.element()
    p.draw(**kwargs)
    return p

def draw_edge(e, **kwargs):
    v1,v2 = e.endpoints()
    l = Line(v1.element(), v2.element())
    l.draw(**kwargs)

#-----------------------------------------------------

def dijkstra(g,src):
    d = {}
    cloud = {}
    pq = AQ.AdaptableHeapPriorityQueue()
    pqlocator = {}
    i=1
    for v in g.vertices():
        if v is src:
            d[v] = 0
            pqlocator[v] = pq.add_task(v,d[v],0)
        else:
            d[v] = float('inf')
            pqlocator[v] = pq.add_task(v,d[v],i+1)
            i+1

    while not pq.is_empty():
        u, key, p = pq.pop_task()
        cloud[u] = key
        del pqlocator[u]
        for e in g.incident_edges(u):
            v = e.opposite(u)
            if v not in cloud:
                wgt = e.element()
                if d[u] + wgt < d[v]:
                    d[v] = d[u] + wgt
                    wgt.draw(width = 3, fill = 'black')
                    pq.update_priorty(pqlocator[v],d[v],v)

    return cloud


if __name__ == "__main__":
    #random_graph("./data/graph1.dat", num_verts=16, min_dist=120, max_dist=250, num_edges=5000)
    #random_graph("./data/graph2.dat", num_verts=24, min_dist=100, max_dist=300, num_edges=5000)
    #random_graph("./data/graph3.dat", num_verts=30, min_dist=80, max_dist=230, num_edges=5000)
    #random_graph("./data/graph4.dat", num_verts=40, min_dist=70, max_dist=200, num_edges=5000)
    #random_graph("./data/graph5.dat", num_verts=40, min_dist=80, max_dist=250, num_edges=5000)
    #random_graph("./data/graph6.dat", num_verts=50, min_dist=50, max_dist=250, num_edges=5000)
    #random_graph("./data/graph7.dat", num_verts=60, min_dist=50, max_dist=160, num_edges=5000)
    #random_graph("./data/graph8.dat", num_verts=80, min_dist=50, max_dist=160, num_edges=5000)
    #random_graph("./data/graph9.dat", num_verts=80, min_dist=50, max_dist=250, num_edges=5000)
    #random_graph("./data/graph10.dat", num_verts=100, min_dist=50, max_dist=250, num_edges=5000)
    #random_graph("./data/graph11.dat", num_verts=80, min_dist=30, max_dist=70, probability=0.6, num_edges=5000)
    #random_graph("./data/graph12.dat", num_verts=80, min_dist=50, max_dist=150, probability=0.85, num_edges=5000)
    g=read_graph("week 9/data/graph1.dat",True)
    v = "220,95"
    c = dijkstra(g,v)
    for i in c:
        print(i.element())
    draw_graph(c)

    show_canvas()


