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

# Differences:

* Dot product

* Broadcasting
   In numpy, broadcasting seems to occur whenever array/matrix sizes are mismatched in an operation, for example:

   ```a = np.array([0,1,2])
      a + 5
   => array([5, 6, 7])
   ```

   in this case an array was added with a scalar.
   
   There are convenience advantages to the numpy approach of automatically determining when to broadcast based on size mismatch, however, the disadvantage is that some types of errors may not be caught. I'm tending to prefer Julia's insistence on using the special broadcasting operator when that's what you want.

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


   

