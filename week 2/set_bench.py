import time
from set_impl_as_list import Set as Setl
from set_impl_as_dict import Set as Setd

def bench_union(SetImpl):
    # A = 10 * list(range(0,8000))
    # B = 10 * list(range(4000, 10000))
    A = 10 * list(range(0,4000))
    B = SetImpl(10 * list(range(2000, 5000)))
    t0 = time.time()
    for i in range(5):
        a = SetImpl(A)
        b = SetImpl(B)
        c = a.union(b)
    t1 = time.time()
    return t1-t0

if __name__ == "__main__":

    list_time = bench_union(Setl)
    dict_time = bench_union(Setd)
    python_set_time = bench_union(set)
    print(f"List implementation union time       = {list_time} seconds")
    print(f"Dictionary implementation union time = {dict_time} seconds")
    print(f"Python set implementation union time = {python_set_time} seconds")