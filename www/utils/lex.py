def find_max(arr):
    i = len(arr)
    j = len(arr[0])
    r = 0
    index = None
    for m in range(i):
        for n in range(j):
            newr = max(r, arr[m][n])
            if newr > r:
                index = m
                r = newr
    return r, index


def longest_sub_sequence(X, Y, m, n):
    # Create a table to store lengths of
    # longest common suffixes of substrings.
    # Note that LCSuff[i][j] contains the
    # length of longest common suffix of
    # X[0...i-1] and Y[0...j-1]. The first
    # row and first column entries have no
    # logical meaning, they are used only
    # for simplicity of the program.

    # LCSuff is the table with zero
    # value initially in each cell
    LCSuff = [[0 for k in range(n + 1)] for l in range(m + 1)]

    # To store the length of
    # longest common substring
    result = 0

    # Following steps to build
    # LCSuff[m+1][n+1] in bottom up fashion
    subs = []
    for i in range(m + 1):
        sub = []
        for j in range(n + 1):
            if i == 0 or j == 0:
                LCSuff[i][j] = 0
            else:
                a = X[i - 1]
                b = Y[j - 1]
                if a == b:
                    LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1
                    result = max(result, LCSuff[i][j])
                else:
                    LCSuff[i][j] = 0
    r = find_max(LCSuff)
    longest = X[r[1] - r[0]: r[1]]
    return longest