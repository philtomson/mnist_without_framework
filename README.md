# MNIST without a framework

The idea here is to take a Python/numpy (no frameworks!)
implementation of MNIST (from: http://jrusev.github.io/post/hacking-mnist/)
and translate it to Julia (also using no framework like Flux or Knet).

Most of numpy's functionality is built-in to Julia. However, numpy's 
linear algebra operations tend to work a bit differently from Julia's.

While the goal is mostly to compare Python/numpy with Julia for implementing
a NN, it's also a learning exercise as frameworks tend to hide a lot of 
details.


# Handy Resources:

*   MATLAB–Python–Julia cheatsheet ( https://cheatsheets.quantecon.org/ )

# Differences between numpy and Julia:

* Dot product, Inner product:

Example dot product and inner product on matrices in Python with numpy:

    >>> a=np.array([[1,2],[3,4]])
    >>> b=np.array([[11,12],[13,14]])
    >>> np.dot(a,b)
    array([[37, 40],
           [85, 92]])
    >>> np.inner(a,b)
    array([[35, 41],
           [81, 95]])


In Julia to get the same results:

    julia> a = [1 2; 3 4]
    julia> b = [11 12; 13 14]
    julia> a*b #equiv to np.dot(a,b)
       2×2 Array{Int64,2}:
       37  40
       85  92
    julia> a*b' #equiv to np.inner(a,b)
       2×2 Array{Int64,2}:
       35  41
       81  95

Note that if you use the dot function you'll get a scalar (which 
seems to make sense):

    julia> import LinearAlgebra
    julia> dot(a,b)
       130

And if you use a broadcasted dot function:

    julia> dot.(a,b)
      2×2 Array{Int64,2}:
      11  24
      39  56    
    

* Broadcasting
   In numpy, broadcasting seems to occur whenever array/matrix sizes are mismatched in an operation, for example:

   ```a = np.array([0,1,2])
      a + 5
   => array([5, 6, 7])
   ```

   in this case an array was added with a scalar.


   In Julia broadcasting needs to be explicitly specified by using a broadcasting operator:

   ```
   julia> a = [0,1,2]
   3-element Array{Int64,1}:
    0
    1
    2
   julia> a+5
   ERROR: MethodError: no method matching +(::Array{Int64,1}, ::Int64)
   julia> a.+5
   3-element Array{Int64,1}:
    5
    6
    7
   ```

   Note that broadcasting operators are prefixed with a '.' as in '.+' above.


   There are convenience advantages to the numpy approach of automatically determining when to broadcast based on size mismatch, however, the disadvantage is that some types of errors may not be caught. I'm tending to prefer Julia's insistence on using the special broadcasting operator when that's what you want.
   

