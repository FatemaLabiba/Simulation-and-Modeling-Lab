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


ASSIGNMENT 3


Student's Assignment 1: Basic Matrix Operations

Create two 3×3 matrices with random integers in Python.
Perform addition, subtraction, and multiplication operations on these matrices.
Compute the determinant, inverse, and rank of one of the matrices

import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix
import seaborn as sns

#1

A = np.random.randint(0, 10, (3,3))
print('A',A)
B = np.random.randint(0, 10,(3,3))
print('B',B)


#2
add = A+B
print('addintion',add)
sub = A-B
print('subtraction',sub)
mul = np.dot(A,B)
print('multiplication',mul)

#3
#choosing matrix A
d = np.linalg.det(A)
print('determinant',d)

if(d !=0 ):
  ind = np.linalg.inv(A)
else:
  ind = 'Determinant is not 0. So, inverse is not possible'
print('inverse',ind)

rank = np.linalg.matrix_rank(A)
print('rank',rank)


Student's Assignment 2: Matrix Plotting and
Visualization
1. Generate and Plot Two Random Vectors:
a. Create two vectors, each containing 15 random floats of your choice.
b. Plot these vectors on the same graph using matplotlib to compare how
they look.
c. Label the axes appropriately and give the plot a title.

#a

A = np.random.rand(15) 
B = np.random.rand(15)
print('A',A,'\nB',B)

#b
x = np.arange(len(A))
plt.plot(x,A, marker = 'o',linestyle = '-', label = 'A')
plt.plot(x,B, marker = '^', linestyle = '--', label = 'B')

#c
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('line graph of matrix')
plt.legend()
plt.grid(True)
plt.show()

2. Generate and Visualize a Random Matrix:
a. Create a 4x4 matrix with random values of your choice.
b. Display the matrix using a heatmap with appropriate color mapping (e.g.,
seaborn or matplotlib).
c. Add labels to the heatmap to indicate row and column numbers.

#a
A = np.random.randint(0, 10, (4,4))
print('A',A)

#b
sns.heatmap(A, annot = True, cmap = 'rocket', linewidths = 0.5, linecolor = 'black')

#c
plt.title('Heatmap of 4*4 Matrix')
plt.show()

3. Matrix Operations and Visualization:
a. Create two 4x4 matrices with random values of your choice.
b. Perform the following operations:
i. Matrix Addition
ii. Matrix Subtraction
iii. Matrix Multiplication (ensure matrices are compatible for
multiplication)

c. For each of these operations, visualize the result using bar plots. Each bar
should represent one element of the matrix.


#a
M = np.random.randint(0, 10, (4,4))
N = np.random.randint(0, 10, (4,4))
print('M',M,'\nN',N)

#b

#i
add = M+N
#ii
sub = M-N
#iii
mul = np.dot(M,N)


#c
fig, axes = plt.subplots(1,3, figsize = (20,5))

#sum
axes[0].bar(range(1, 17), add.flatten(), color = 'b')
axes[0].set_xlabel('Index')
axes[0].set_ylabel('Value')
axes[0].set_title('Matrix_Addition')

#substraction
axes[1].bar(range(1,17),sub.flatten(),color = 'r')
axes[1].set_xlabel('Index')
axes[1].set_ylabel('Value')
axes[1].set_title('Matrix_substraction')

#multiplication
axes[2].bar(range(1,17),mul.flatten(),color = 'k')
axes[2].set_xlabel('Index')
axes[2].set_ylabel('Value')
axes[2].set_title('Matrix_substraction')



