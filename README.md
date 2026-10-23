# Task 14: NumPy Matrix Operations

## Description
This task demonstrates fundamental matrix operations using Python and the NumPy library. Understanding these basic operations—such as addition, subtraction, matrix multiplication, and transposition—is crucial for data manipulation and building machine learning algorithms.

## Objective
To understand and implement basic matrix operations used in machine learning environments.

## Tools Used
* Python
* NumPy
* Jupyter Notebook (or any standard Python IDE)

## Prerequisites
Ensure you have NumPy installed in your Python environment. You can install it via pip:
```bash
pip install numpy
```

## Operations Covered
1. **Matrix Creation:** Using `np.array()` to define 2D arrays.
2. **Element-wise Addition & Subtraction:** Performing operations on matrices of the exact same shape (`+` and `-`).
3. **Matrix Multiplication (Dot Product):** Multiplying matrices where the inner dimensions match using `np.dot()` or the `@` operator.
4. **Transpose:** Flipping a matrix over its diagonal using the `.T` attribute to switch row and column indices.

## Expected Output
When running the accompanying script or notebook cells, you should see the following console output:

```text
Matrix A:
 [[1 2 3]
 [4 5 6]]

Matrix B:
 [[ 7  8]
 [ 9 10]
 [11 12]]

Matrix C (Square matrix for element-wise demo):
 [[1 2]
 [3 4]]

Addition (C + D):
 [[ 6  8]
 [10 12]]

Subtraction (C - D):
 [[-4 -4]
 [-4 -4]]

Matrix Multiplication (A @ B):
 [[ 58  64]
 [139 154]]

Original Matrix A (2x3):
 [[1 2 3]
 [4 5 6]]

Transposed Matrix A (3x2):
 [[1 4]
 [2 5]
 [3 6]]
```

## Interview Questions & Answers

**1. What is a matrix?**
A matrix is a two-dimensional rectangular array or grid of numbers, symbols, or expressions arranged in rows and columns. In programming and data science, it is commonly represented as a 2D array.

**2. What is the difference between element-wise and matrix multiplication?**
Element-wise multiplication multiplies numbers at the exact same corresponding positions in two matrices of the same shape. Matrix multiplication (dot product) multiplies the rows of the first matrix by the columns of the second matrix, requiring the inner dimensions to match (e.g., a 2x3 matrix multiplied by a 3x2 matrix yields a 2x2 matrix).

**3. Why are matrices important in machine learning?**
Matrices are the fundamental structure for representing datasets (rows as samples, columns as features). They allow for efficient, vectorized mathematical computations across massive amounts of data simultaneously, which is crucial for operations like updating weights in neural networks and performing linear algebra transformations in algorithms.
