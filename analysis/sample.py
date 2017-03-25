from scipy.sparse import dok_matrix

a = dok_matrix((3, 10))

a[1, 2] = 2
a[2, 2] = 2
a[2, 3] = 1
a[2, 4] = 3

print(a[2].nonzero()[1])
print(a[2].nonzero())
print(a[2].values())
