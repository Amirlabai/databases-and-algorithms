import random
import time
import matplotlib.pyplot as plt
import numpy as np

def bubble_sort_runtime_graph():
#    Size =
    Time = list()
    for N in Size:
#        t =
        t = round(t,4)
        Time.append(t)
    plt.plot(Size,Time)
    plt.xlabel('List Size')
    plt.ylabel('Run Time')
    plt.title('run time/list size')
    plt.show()