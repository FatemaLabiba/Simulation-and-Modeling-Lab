# Simulation-and-Modeling-Lab
All code of simulation and modeling lab 

Lab Assignment 2:

1. Create Your Personal Matrix
Let:
● d1 = last digit of your student ID
● d2 = second last digit of your student ID
Create the matrix:
A = [[D1+2,d2+1],[2d1, d2+2]]
d1 = 3
d2 = 1
A = [[5,2],[6,3]]
import numpy as np
from sympy import Matrix
import matplotlib.pyplot as plt
import seaborn as sns
2. Analyze Matrix A
Using NumPy, compute:
● Shape of A
● Determinant
● Rank
● Eigenvalues
● Inverse (only if determinant ≠ 0)
A = np.array([[5,2],[6,3]])

#shape of A
print('Shape of A is',A.shape)

#determinant
det_A = np.linalg.det(A)
print('Determinant of A = ',det_A)

#Rank
rank_A = np.linalg.matrix_rank(A)
print('Rank of A is',rank_A)

#Eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A)
print('Eigenvalue',eigenvalues)

#Inverse
if det_A != 0:
  print('Inverse of A',np.linalg.inv(A))
else:
  print('Matrix is singular, cannot compute inverse')

3. Value Change Experiment
Create a new matrix B by changing exactly one value in A:
● Add +1 to one element of your choice
● You must clearly state which element you changed

4. Re-analyze Matrix B
Compute the same properties as in Step 2.

Step 2 is
Using NumPy, compute:
● Shape of A
● Determinant
● Rank
● Eigenvalues
● Inverse (only if determinant ≠ 0)

Maritx A = ([[5,2],[6,3]])

B = np.array([[5,2],[6,4]])

#Shape of B
print("Shape of B",B.shape)

#determinant
print('Determinant',np.linalg.det(B))

#Rank
print('Rank of B',np.linalg.matrix_rank(B))

#Eigenvalue
eigenvalues,eigenvactors = np.linalg.eig(B)
print('Eigenvalues',eigenvalues)

#Inverse
if(np.linalg.det(B) != 0):
  print('Inverse of B',np.linalg.inv(B))
else:
  print('Matrix is singular, cannot compute Inverse')
Comparison & Explanation
Answer the following in your own words:
  1. How did the determinant change and why?
  2. Did the rank change? Explain.
  3. How did the eigenvalues respond to the value change?
  4. Is B easier or harder to invert than A? Why?

1. The determinant of A=2.9999 and B = 7.9999 . Here we can see the value of A and B are not same. Because we know the equation to find determinant is det = [ad-bc] as I changed one value in matrix B from earlier matrix A, so the value of determinant will not be same for A and B.

2. No, the rank for both A and B were 2. If determinant is not equal to 0 then rank will be 2(full rank) else rank will be 1 or 0 for 2*2 matrix. As A and B both matrix's determinant is not equal to 0 then the rank will remain the same.

3. As I changed one value in Matrix B from earlier matrix A, eigenvalue is also changed. Eigenvalue is depends on the element inside matrix. So eigenvalue is changing due to the change of element.

4. B is easier to invert than A. Because the determinant value of B matrix is greater then A matrix.

