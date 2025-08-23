import numpy as np

def levenshtein(s1 : str, s2 : str):
    m, n = len(s1), len(s2)
    d = np.zeros((m+1, n+1), dtype=int)

    for i in range(m+1):
        d[i, 0] = i
    for j in range(n+1):
        d[0, j] = j

    cost = 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost += 1 if s1[i-1] == s2[j-1] else 0
            d[i, j] = min(cost, d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + cost) # Deletion, insertion, substitution

    return d[m, n]

from rapidfuzz import fuzz

string1 = 'Spiderman'
string2 = 'Superman'

print(fuzz.ratio(string1, string2))
print(levenshtein(string1, string2))