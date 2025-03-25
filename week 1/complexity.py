import random
import time
import matplotlib.pyplot as plt
import numpy as np
import math

def prime(x=10000):
    start_time = time.time()
    for i in range(2,x):
        
        bo = True
        if i % 2 == 0:
            bo = False
        else:
            for j in range(2,int(math.sqrt(i))):
                if (j % 2) != 0:
                    if (i % j) == 0:
                        bo = False
                        break
        if bo:
            end_time = time.time()
    return end_time - start_time
    

def bubble_sort_runtime_graph():
    Size = [100,1000,10000]
    Time = list()
    for N in Size:
        t = prime(N)
        t = round(t,4)
        Time.append(t)
    plt.plot(Size,Time)
    plt.xlabel('List Size')
    plt.ylabel('Run Time')
    plt.title('run time/list size')
    plt.show()

bubble_sort_runtime_graph()