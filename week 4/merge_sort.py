# L stands for any mutable object that has an array interface
# Like a standard Python list for example
# For simplicity, L is assumed to be a list of integers, but the algorithm
# applies to any object that also implements the comparison operators: '<',
# '>', '==', '<=', '>='

def merge_sort(L):
    merge_sort_rec(L, 0, len(L) -1)

# merge sort recursive (used by merge_sort)
def merge_sort_rec(L, first, last):
    if first < last:
        sred = (first + last)/2
        merge_sort_rec(L, first, sred)
        merge_sort_rec(L, sred + 1, last)
        merge(L, first, last, sred)

# merge (used by merge_sort_rec)
def merge(L, first, last, sred):
    helper_list = []
    i = first
    j = sred + 1

    while i <= sred and j <= last:
        if L[i] <= L[j]:
            helper_list.append(L[i])
            i += 1
        else:
            helper_list.append(L[j])
            j += 1

    while i <= sred:
        helper_list.append(L[i])
        i +=1

    while j <= last:
        helper_list.append(L[j])
        j += 1

    for k in range(0, last - first + 1):
        L[first + k] = helper_list [k]