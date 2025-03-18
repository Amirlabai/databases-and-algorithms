import random
import time
import matplotlib.pyplot as plt
import numpy as np

def bubble_sort_runtime_graph():
    Size = [0,110,202,1,4,6]
    Time = list()
    for N in Size:
        t = N
        t = round(t,4)
        Time.append(t)
    plt.plot(Size,Time)
    plt.xlabel('List Size')
    plt.ylabel('Run Time')
    plt.title('run time/list size')
    plt.show()

bubble_sort_runtime_graph()