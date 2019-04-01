import numpy as np

def d(x1,x2):
    return abs(x1-x2)


def dtw_low_space(a1, a2, bound=100, d=d):
    assert(len(a1) == len(a2))
    mid = bound + 1
    m = np.full((2,2*bound+3), np.infty)
    # m[0,mid] = d(a1[0], a2[0])
    m[1, mid] = 0
    paths = [None for _ in range(2*bound+3)]
    paths[mid] = (None, (-1,-1, 0))
    for i in range(0,len(a1)):
        k = i % 2
        l = 1-k
        for j in range(max(mid-i, 1), min(mid+(len(a1)-i-1), mid+bound)+1):
            (val, rel) = min(zip((m[k, j - 1], m[l, j], m[l, j + 1]), (0, 1, 2)))
            diff = a1[i] - a2[i-mid+j]
            m[k, j] = val + d(a1[i], a2[i-mid+j])
            prev = paths[j+rel-1]
            offset = ((0, 1), (1,1), (1, 0))[rel]
            paths[j] = (prev, (prev[1][0] + offset[0], prev[1][1] + offset[1], diff))
    path = []
    next = paths[mid]
    while (next[0] is not None):
        path.append(next[1])
        next = next[0]
    path.reverse()
    return m[(len(a1)-1) % 2, mid], path


def multi_dtw_low_space(tss1, tss2, bound, d=d):
    assert(len(tss1) == len(tss2))
    total = 0
    for i in range(len(tss1)):
        total += dtw_low_space(tss1[i], tss2[i], bound, d) # ** 2
    return total


if __name__ == '__main__':
    a1 = np.array([6,3,4,2])
    a2 = np.array([3,8,6,-2])
    a3 = np.array([1,5,7,8])
    print(dtw_low_space(a1, a2, 2))
    print(dtw_low_space(a2, a3, 2))
    print(dtw_low_space(a1, a3, 2))


