# Variable-Projection-Julia

This is a WIP, but I have applied the code to a number of fitting applications already.

Variable Projection is a method published in the 1960s by Golub and Pereyra that offers a solution to separable nonlinear minimization problems of the form:

minimize β for f = ( L(M(β)α,y) ), where:

* L(*) is the least squares loss function,
* M(*) is a nonlinear basis parameterized by β,
* α is a vector of unknown scaling coefficients
* y is the data being fit

The parameters which enter the model linearly are separated out from the nonlinear ones and solved via linear least squares.  The result is that the nonlinear optimizer only "sees" the nonlinear variables.  This problem formulation is particularly well suited to fitting sums of lineshape functions.
