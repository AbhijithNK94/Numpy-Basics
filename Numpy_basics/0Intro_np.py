"""
numpy:-- Numerical Python
numpy, pandas, scipy are libraries used for data exploration and analysis.
numpy provides linear algebra and statistical computing fns, mainly for numerical ops.
Numpy, Scipy are used for:
                         :- performing basic scientific and mathematical ops.

Data is stored in numpy as 'ndarray' objects.
"""
"""
==> LIST vs ARRAY

=> Similarities:
Internally it contains lists.
Array is same like list to store data.
Both are mutable, indexed.
Both can be sliced/slicing operation.

=> Differences:
List contain diff data types, but array can have elements of same data types.
We uses array mainly due to need less memory than list, faster than list, convenient to use.
We can perform ops(+ * etc) directly to an array. But we cannot do this on list.
For Numpy we want to install Numpy, but list is built_in.

Commonly used abbreviations:
NumPY  - np
Pandas - pd
"""

import numpy as np  # It is preferable to use the abbr np for importing numpy always.
import sys
# s = range(100)  # list
# print(s)
# print(sys.getsizeof(s))
# print(sys.getsizeof(s)*len(s))  # total size = size of one element * total element
#
# A = np.arange(100)  # to create an array
# print(A)
# print(A.size)
# print(A.itemsize)
# print(A.size*A.itemsize)

"""One dimensional array"""
# n_array = np.array([12, 45, 78])  # creating an array
# print(n_array)
# print(type(n_array))  # checking the type of a variable.
# print(n_array.ndim)  # to get the dimension of array.
# print(n_array.dtype)  # type of array/data stored in the array eg: float64 ==> 64bits => 64/8 ==> 8 bytes
# # here we got output as int32 ==> 32bits ==> 32/4 ==> 4 bytes
# print(n_array.shape)  # how many elements present in each dim (columns,) or (rows, columns).
# print(n_array.size)  # total number of elements.
# print(n_array.data)  # memory address.
# print(n_array.itemsize)  # Will gives the bytes required to store the data.

"""Two Dimensional array"""
# twoD = np.array([[10, 20, 30], [40, 50, 60]])
# print(twoD)
# print(twoD.ndim)
# print(twoD.shape)
# print(twoD.dtype)
# print(twoD.size)  # total number of elements in the matrix or array.
# print(twoD.itemsize)  # memory usage of the stored data twoD.
#
# narray = np.array([12, 45, 78], dtype='int64')
# to change the data type as stored

"""
Methods to create an array:
=> array()
=> arange()
=> ones()
=> zeroes()
=> eye()
=> linspace()
=> random()
"""

# 1) array() - create array from list and tuples.
# a = np.array([1, 2, 3])
# print(a)
# a = np.array((1, 2, 3))  # the type of collection of arg that is passed on does not affect the creation of array.
# print(a)
# print(a.ndim)
# print(a.dtype)  # by default it takes data type as min type req to hold the obj.
#
# narray = np.array([12, 45, 78], dtype='int64')
# print(narray.dtype)

# 2D
# a = np.array([[1, 2], [3, 4]])
# print(a)
# print(a.ndim)

#3D
# a = np.array([[[1, 2, 2], [3, 4, 4]], [[5, 6, 6], [7, 8, 8]]])
# print(a)
# print(a.ndim)  # 3d array/ cube/ 3d tensor
# print(a.shape)

# arange() - create an array of evenly spaced values
# same like range fn, but return an ndarray
# z = np.arange(10)
# print(z)
# z = np.arange(5.0)
# print(z)
# z = np.arange(2, 10)
# print(z)
# z = np.arange(2, 10, 2)
# print(z)
# z = np.arange(10, dtype='complex')
# print(z)

# 3) zeros() - create array filled with zeros
# shape should be given as int or tuple of int
# default type is float
# s = np.zeros(5)
# print(s)
# s = np.zeros((3, 4))  # shape of array should be passed as an arg in tuple form
# print(s)

# 4) Ones() - create an array filled with ones
# default type is float
# r = np.ones(4)
# print(r)
# r = np.ones(4, dtype='int')  # we can change the def data type (float) to int by passing dtype as an arg.
# print(r)
# r = np.ones((4, 4, 2))  # 4 matrix of size 4X2
# print(r)

# 5) linspace() - create array filled with evenly spaced values.
# similar to arange fn, but gives end point also
# a = np.linspace(2, 10) # start and end points
# print(a)  # we get 50 points by default.

# b = np.linspace(1, 10, 4)
# print(b)
# # exclude last point
# b = np.linspace(2, 10, 4, endpoint=False)
# print(b)
#
# # 6) eye () - gives an array filled with zeros except k-th diagonal to 1.
# a = np.eye(3)  # n = m by def (rows and columns)
# print(a)
# """We can define where we want to locate the one's with ref to principal diagonal elements"""
# # define where we want 1s.
# a = np.eye(3, k=1)
# print(a)
# a = np.eye(3, k=2)
# print(a)
# a = np.eye(3, k=-1)
# print(a)
# a = np.eye(3, k=-2)
# print(a)

# same to eye() we have identity fn for sqr matrix
# a = np.identity(3)
# print(a)

# random() - for creating random number array
# fns in random module ==>
# -rand()
# -randn()
# -randint()

# a = np.random.rand(5)  # uniformly distributed 5 random numbers.
# print(a)
#
# a = np.random.rand(5, 2)  # uniformly distributed random numbers (5, 2) is the shape of array.
# print(a)

# b = np.random.randn(4)  # normal distributed numbers (Gaussian distribution)
# print(b)
#
# c = np.random.randint(2, 6)  # gives a random number b/w 2 and 6
# print(c)
#
# c = np.random.randint(100, size=4)  # 4 random numbers
# print(c)
#
# c = np.random.randint(100, size=(2, 3))  # creates an 2 D array of 2 rows and 3 columns (2, 3) passed as an arg to the
# #                                            size
# print(c)

"""Numerical data types:
Boolean, integer, unsigned, integer, float, complex

bool - true, false
int8 - Byte(-128 to 127)
int16 - Integer(-32768 to 32767)
int32 - Integer(-9223372036854775808 to 9223372036854775807)
# int64          Integer (-9223372036854775808 to 9223372036854775807)
# uint8          Unsigned integer (0 to 255)
# uint16         Unsigned integer (0 to 65535)
# uint32         Unsigned integer (0 to 4294967295)
# uint64         Unsigned integer (0 to 18446744073709551615)
# intp           Integer used for indexing, typically the same as ssize_t
# float32        float
# float64
# complex64      Complex number, represented by two 32-bit floats (real and imaginary components)
# complex128

i - integer
f - float
? - bool
U - unicode
"""
# a = np.array([True, False, True])
# print(a)
# print(a.dtype)
#
# a = np.array([1+2j, 3-6j])
# print("\n", a)
# print(a.dtype)
#
# a = np.array(['apple', 'banana', 'cherry'])
# print('\n', a)
# print(a.dtype)
#
# a = np.array([8, 2, 3])
# print('\n', a)
# print(a.dtype)
#
# a = np.array([214783648, 25435545, 3])
# print('\n', a)
# print(a.dtype)
#
# a = np.array([2147836489, 25435545, 3])
# print('\n', a)
# print(a.dtype)

# a = np.array([8, 0, -3], dtype='bool')
# print(a)
# print(a.dtype)
#
# a = np.array([8, 2, 3], dtype='f')
# print(a)
# print(a.dtype)
#

"""NB: For the uint8, int8, whatever number is being given if the number is within the range,
then the o/p will be the number itself and if the number exceeds or precedes the limit of 
data types, then it will convert to the specific data type by looping method.
For eg: uiint range(0-255)    if the i/p is 256 then the no: exceeds the range by 1.
                              Hence it will go back to the start and o/p will be 0."""
# a = np.array([-5, 2, 3, 100, -100], dtype='uint8')
# print(a)
# print(a.dtype)

# a = np.array([128, 2, 3], dtype='int8')
# print(a)

# b = np.array([[1, 2], [3, 4]])
# print(b)
# print(b[0][0])
# print(b[0][1])
# print(b[1][1])
# print(b[-1][-1])
# print(b[-1][-2])
# print(b[-2][-2])
# print(b[-2][-1])

# 3D array in a -2D array
# 2 Dimensional matrix in a 3 Dimensional matrix.

# c = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]
# d = np.array(c)
# print(d)
# print(d.ndim)
# print(d[0][1][1])  # matrix, row, column
# print(d[1][0][0])
# print(d[1][2][3])
# print(d[-2][-3][-4])
# print(d[-1][-3][-4])
# print(d[0])
# print(d[0][1])
#
# print(d[0, 1, 1])
# print(d[0, :, 1])  # Slicing of a column of a matrix. (":" in a list or array represents the span)
# print(d[0, :, :])  # represents all rows and columns of 0 the matrix
# print(d[0, 1:, 1:])  # displays all elements except 0th row and column.
# print(d[0, 1:, 0])

"""
                         CONCEPT OF 1D, 2D & 3D ARRAYS.
       
==> For a 1D array: there will only be one position: eg: a = [1, 2, 3] have only one position 
    to specify for identifying the elements of array.
a[0] = 1
a[1] = 2
a[2] = 3
==> For a 2D array: we need to specify 2 positions inorder to locate the elements of array.
eg: a = [[1, 2, 3],
         [3, 1, 2],
         [1, 0, 2]] is a 2D array. In this for locating an element in the array, we need to
         specify 2 positions (row and columns).
         a[0] = [1, 2, 3] giving one pos will give us an array or row of that pos.
         a[0][0] = 1 giving 2 pos will give us the element of 2D array.
         a[0][1] = 2
==> For a 3D array: we need to specify 3 positions for locating the elements of 3D array.
eg: a = [[[1, 2, 3],
          [2, 0, 1],
          [1, 3, 1]],
         [[1, 5, 0],
          [2, 1, 0],
          [1, 5, 6]]] is a 3D array. In this for locating an element in the array, we need 
          to specify 3 positions. ie; (matrix, row, column)
          a[0] = [[1, 2, 3],
                  [2, 0, 1],
                  [1, 3, 1]] will give us the 2D array/matrix of 0th pos.
          a[0][0] = [1, 2, 3] will give us the 1D array of 0th pos matrix's 0th row/array.
          a[0][0][0] = 1 will give us the 0th element of 0th row/1D array of 0th matrix/2D array
"""

# Assign values:
# Array is mutable similar to that of list.
# a = np.array([1, 2, 3, 4, 5])
# print(a)
# a[3] = 100
# print(a)
#
# b = np.array([[1, 2], [3, 4]])
# print(b)
# b[0][1] = 100
# print(b)

# d[0][1][1] = 100
# print(d)
#
# d[0, 1:, 1:] = 111
# print(d)

# d = np.arange(10)
# print(d)
# d[5:] = 10
# print(d)
# b = np.arange(4)
# print(b)
# print(b[::-1])
# d[6:] = b[::-1]  # Here we are assigning all the values of array b in reversed form from the 6th pos.
# print(d)

# a = np.array([8, 2, 5, 9, 11, 6])
# print(a)
# print(a[:])  # represents all the elements of array
# print(a[2:])  # represents all the elements of array starting from 2nd position
# print(a[2:5])  # represents all the elements of array starting from 2nd pos to 5th pos (5th pos element excluded)
# print(a[:4])  # represents all the elements from zeroth pos up to 4th pos (4th pos ele excluded)
# print(a[0:6:2])  # represents all the elements starting from 0th pos to 6th pos with step size 2.
# print(a[::-1])  # represents the array in rev form

# 2D array = x[srt:end:stp, srt:end:stp]
#            x[rows, columns]
# x = np.array([[1, 2], [3, 4], [5, 6]])
# print(x)
# print(x[1:, 1:])
# print(x[1, :])
# print(x[1:])

# y = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [20, 30, 40, 50]])
# print(y)
# print(y[1:, 1:])
# print(y[1, 1])
#
# print(y[::, ::2])

# 3D array ==> x[srt:end:stp, srt:end:stp, srt:end:stp] ie: (matrix, rows, columns)
# z = [[[1, 2, 3, 4],
#      [5, 6, 7, 8],
#      [9, 10, 11, 12]],
#      [[13, 14, 15, 16],
#       [17, 18, 19, 20],
#       [21, 22, 23, 24]]]
# a = np.array(z)
# print(a, '\n')
# print(a[1:], '\n')  # represents the matrix from 1st position to the last pos.
# print(a[0:], '\n')  # represents the matrix from zeroth pos to last pos
# print(a[0:1])  # represents the matrix at zeroth pos.
# print(a[1:, 1:, 1:])  # represents the matrix at pos 1, rows and columns except 0th pos of that matrix.


# Scalar and vector addition and subtraction.
# a = np.array([10, 20, 90, 90, 70])
# b = np.ones(5, dtype='int') + 1
# print(b)
# print(a - b)
# NB: if the size and shape of arrays are different, then the algebraic ops are not possible.

# a1 = np.array([[1, 2], [3, 4]])
# a2 = np.array([[2, 3], [1, 2]])
# print(a1)
# print(a2)
# print('mul', a1*a2)  # Matrix multiplication not possible with the * operator. '*' operator
# #                      only does the element multiplication of 2 arrays.
# print('add', a1+a2)
# print('mul', a1.dot(a2))  # actual matrix multiplication

# Transcendental fns (a fn not expressible as a finite)
# Combination of the algebraic ops of addition, sub, mul
# a = np.arange(5)
# print(a)
# print(np.sin(a))
# print(np.log(a))
# print(np.exp(a))
# print(np.sqrt([2, 5, 8]))

# Shape mismatch
# a = np.arange(5)
# b = np.arange(2)
# print(a + b)  # Will give us an error due to shape mismatch.

# Basic reductions ( Ops of elements within the array)
# arr = np.array([10, 20, 90, 90, 70])
# print(arr.sum())
# print(np.sum(arr))
# print(arr.max())
# print(np.max(arr))
# print(arr.min())
# print(np.min(arr))
# print(arr.argmin())  # index/pos of min ele
# print(arr.argmax())  # index/pos of max ele

# x = np.array([[1, 1], [2, 2]])
# print(x)
# print(x.sum())
# print(x.sum(axis=0))  # column wise operation
# print(x.sum(axis=1))  # row wise operation

# Statistics using NumPY
# x = np.array([1, 2, 3, 1])
# print(x.mean())  # Its the avg value, it change even one value changes
# print(np.median(x))  # It gives central value from sorted order, it will not affect much if any value changes lot.
# print(x.std())
# print(np.std(x))

"""Matrix ops using NumPY"""
# Transpose of a Matrix:
# x = np.array([[1, 1, 3], [2, 2, 5]])
# print(x)
# print(x.T)

# Flattening
# print(x.ravel())  # Converts a multidimensional array to a 1D array
# print(np.ravel(x))

# Reshape
# a = np.arange(24)
# print(a)  # 1D array
# print(a.reshape((6, 4)))  # reshaping to a 2D array of 6 rows and 4 columns
# print(a.reshape((6, 2, 2)))  # reshaping to a 3D array of 6 matrices, 2 rows and 2 columns.
"""
NB : While passing the args for reshaping, the number of elements(size) of 1D array should be
equal to the product of the shape of 2D/3D array.
ie; size (1D array) = no.of rows * no.of columns (2D array)
    size (1D array) = no.of matrix * no.of rows * no.of columns (3D array)
"""

# Resizing  # fill with zero
# a = np.arange(4)
# print(a)
# a.resize(8)
# print(a)
"""NB: If we referenced to some other variable, resizing will not work"""

# Sorting
# a = np.array([[5, 4, 6], [2, 3, 2]])
# print(a)
# b = np.sort(a)
# print(b)
# b = np.sort(a,axis=0)
# print(b)

# a.sort(axis=1)  # row wise sorting of array a and save in itself.
# print(a)
# a.sort(axis=0)
# print(a)

# Arg sorting
# a = np.array([6, 8, 5, 2, 9, 1, 3])
# b = np.argsort(a)  # Gives us the index pos of sorted elements
# print(b)


a = [10, 20, 40, 80, 20, 60, 50, 90, 100, 110, 250, 112, 10]
print(np.percentile(a, 25))
print(np.percentile(a, 50))
print(np.percentile(a, 75))
print(np.percentile(a, 90))
"""
   Quartiles are 1st quartile = 25%
                 2nd quartile = 50%
                 3rd quartile = 75%
Percentile of a data in a dataset implicates the percentage of total number of data's to the total number of data's 
which comes below or equal to the data under observation.
Percentile of a data in a dataset is given by the eqn:
n = (P/100) * N
where n ==> index/pos of data in the sorted data list.
      N ==> Size of the dataset / No: of elements present in the dataset.
      P ==> Percentile
"""
# Example: Given below will be a dataset of marks of 20 students for a max mark of 50.
mark_list = [45, 32, 48, 36, 35, 37, 41, 40, 48, 30, 36, 39, 36, 43, 45, 38, 47, 42, 41, 35]
# Sorting the mark_list will give
sorted_list = np.sort(mark_list)
print(sorted_list)
# O/P = [30 32 35 35 36 36 36 37 38 39 40 41 41 42 43 45 45 47 48 48]
# Suppose we wanna get the percentile of 36 in the given mark_list.
N = len(mark_list)
n = max(max(np.where(sorted_list == 36)))
print(n)
Percentile = (n * 100) / N
print(Percentile)  # O/P = 30
# We get the percentile as 30, which means 30% of the marks in the mark_list are less than or equal to 36 (<=36)
# Percentile calculation using NumPY.
print(np.percentile(mark_list, 30))  # O/P = 36
"""
Here for the percentile fn, the args that we need to pass on is 'np.percentile(array_name, percentile)'.
The O/P will be the data that will be equal to or greater than the percentage number of data's to the total number of 
data's or percentile.
"""




