import math
import time
from tkinter import messagebox
import matplotlib.pyplot as plt

Size = []
Time = []
x=10000

def prime(x=10000):
    for i in range(2,x):
        start_time = time.time()
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
            Size.append(i)
            end_time = time.time()
            elapsed_time = end_time - start_time
            Time.append(elapsed_time)


#messagebox.showinfo("", f"Elapsed time: {elapsed_time:.6f} seconds")
prime()

plt.plot(Size,Time)
plt.xlabel('List Size')
plt.ylabel('Run Time')
plt.title('run time/list size')
plt.show()