import numpy as np

# Creating NumPy arrays
array1d = np.array([1, 2, 3, 4, 5])
array2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Basic array information
print("Shape of array2d:", array2d.shape)
print("Number of dimensions in array2d:", array2d.ndim)
print("Number of elements in array2d:", array2d.size)
print("Data type of array2d:", array2d.dtype)

# Array indexing and slicing
print("Element at row 1, column 2:", array2d[1, 2])
print("First row:", array2d[0, :])
print("Second column:", array2d[:, 1])
print("Slicing a subarray:", array2d[0:2, 1:3])

# Mathematical operations
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

sum_result = np.add(array1, array2)
difference_result = np.subtract(array1, array2)
product_result = np.multiply(array1, array2)
division_result = np.divide(array1, array2)
dot_product = np.dot(array1, array2)

print("Sum:", sum_result)
print("Difference:", difference_result)
print("Product:", product_result)
print("Division:", division_result)
print("Dot Product:", dot_product)

# Statistical operations
mean = np.mean(array1)
median = np.median(array1)
std_deviation = np.std(array1)
variance = np.var(array1)

print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std_deviation)
print("Variance:", variance)

# Element-wise operations
squared = np.square(array1)
square_root = np.sqrt(array1)
exponential = np.exp(array1)

print("Squared:", squared)
print("Square Root:", square_root)
print("Exponential:", exponential)

# Concatenation
concatenated = np.concatenate((array1, array2))
print("Concatenated array:", concatenated)

# Generating sequences
sequence = np.arange(0, 10, 2)
print("Sequence:", sequence)

# Element-wise comparison
greater_than_3 = array1 > 3
print("Elements greater than 3:", array1[greater_than_3])

# Boolean operations
any_true = np.any(array1 > 2)
all_true = np.all(array1 > 2)
print("Any element greater than 2:", any_true)
print("All elements greater than 2:", all_true)

# Sorting
sorted_array = np.sort(array1)

# More NumPy functions

# Element-wise absolute value
abs_values = np.abs(array1)

# Element-wise rounding
rounded = np.round(array1, decimals=2)

# Element-wise power
powered = np.power(array1, 2)

# Element-wise sine, cosine, tangent
sine_values = np.sin(array1)
cosine_values = np.cos(array1)
tangent_values = np.tan(array1)

# Element-wise logarithm
logarithm_base_10 = np.log10(array1)

# Element-wise exponentiation
exponential_base_2 = np.exp2(array1)

# Matrix multiplication (dot product)
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
matrix_product = np.matmul(matrix1, matrix2)

print("Absolute Values:", abs_values)
print("Rounded Values:", rounded)
print("Powered Values (element-wise):", powered)
print("Sine Values:", sine_values)
print("Cosine Values:", cosine_values)
print("Tangent Values:", tangent_values)
print("Logarithm Base 10:", logarithm_base_10)
print("Exponential Base 2:", exponential_base_2)
print("Matrix Product:")
print(matrix_product)

# Using np.where
condition = array1 > 2
where_result = np.where(condition, 'A', 'B')
print("np.where Result:", where_result)

# Using np.intersect1d
intersection_result = np.intersect1d(array1, array2)
print("Intersection Result:", intersection_result)

# Advanced Functions

# Matrix Determinant
matrix_determinant = np.linalg.det(matrix1)
print("Matrix Determinant:", matrix_determinant)

# Matrix Inverse
matrix_inverse = np.linalg.inv(matrix1)
print("Matrix Inverse:")
print(matrix_inverse)


