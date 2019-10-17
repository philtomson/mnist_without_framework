# MNIST without a framework

The idea here is to take a Python/numpy (no frameworks!)
implementation of MNIST (from: http://jrusev.github.io/post/hacking-mnist/)
and translate it to Julia (also using no framework like Flux or Knet).

Most of numpy's functionality is built-in to Julia. However, numpy's 
linear algebra operations tend to work a bit differently from Julia's.

While the goal is mostly to compare Python/numpy with Julia for implementing
a NN, it's also a learning exercise as frameworks tend to hide a lot of 
details.

#Handy Resources:

*   MATLAB–Python–Julia cheatsheet ( https://cheatsheets.quantecon.org/ )


