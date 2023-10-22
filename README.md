# tensorgrad
 An autograd engine using tensors/matrices instead of scalar values. 
 Due to Python's GIL, it is impossible to gain the benefit of using matrix transforms to parallelise the backward pass without creating a c extension.
 Mainly for educational purposes
