# problem 1
def remove_dups(l):
    preset_list = set()
    new_list = []
    for i in l:
        if i not in preset_list:
            preset_list.add(i)
            new_list.append(i)
    return new_list

# problem 2
def has_dup(l):
    present = set()
    for i in l:
        if i in present:
            return True
        present.add(i)
    return False     

# problem 3
def is_element_sum(sum,list_a,list_b):
    set_b = set(list_b)
    for i in list_a:
        if sum - i in set_b:
            print(f"{sum} = {i} + {sum-i}")

# problem 3b
def is_element_sum_b(sum, list_a, list_b):
    ret = False
    i = 0
    j = len(list_b)-1
    while i < len(list_a) and j >= 0:
        if sum == list_a[i]+list_b[j]:
            print(f"{sum} = {list_a[i]} + {list_b[j]}")
            i+=1
            j-=1
            ret = True
        elif sum > list_a[i]+list_b[j]:
            i+=1
        else:
            j-=1
    return ret

# problem 4
def radix_sort(L):
    RADIX = 10
    deci = 1
    while True:
        buckets = [list() for i in range(RADIX)]
        done = True
        for n in L:
            q = n / deci # q = quotient
            r = q % RADIX
            # r = remainder = last digit
            buckets[r].append(n)
            if q > 0:
                done = False
            # i has more digits
        i = 0 # Copy buckets to L (so L is rearranged
        for r in range(RADIX):
            for n in buckets[r]:
                L[i] = n
                i += 1
        if done: break
        deci *= RADIX

        
L = [
    7,2,2,5,7,2,1,7,3,10000
]

print(L)
print(has_dup(L))

M = remove_dups(L)

print(M)
print(has_dup(M))

is_element_sum(10,M,M)
M.sort()
print(is_element_sum_b(10,M,M))