# Variable-Projection-Julia

This is a WIP, but I have applied the code to a number of fitting applications already.

Variable Projection is a method published in the 1960s by Golub and Pereyra that offers a solution to separable nonlinear minimization problems of the form:

minimize β for f = ( L(M(β)α,y) ), where:

* L(*) is the least squares loss function,
* M(*) is a nonlinear basis parameterized by β,
* α is a vector of unknown scaling coefficients
* y is the data being fit

The parameters which enter the model linearly are separated out from the nonlinear ones and solved via linear least squares.  The result is that the nonlinear optimizer only "sees" the nonlinear variables.  This problem formulation is particularly well suited to fitting sums of lineshape functions.  In my implementation of this idea, I use NLopt to solve the nonlinear problem, and so that package (which is fairly stable) is required.  At present I don't offer flexibility for using other solvers.

## How to use:

Usage revolves around 3 main types:
1. `ModelParams`: Instantiating this object fixes a bunch of things that the solvers will need.  You must provide it with (in order of arguments the constructor expects):
    a.  An Array of `BoundedLineShape`.  The nonlinear bounds for each `LineShape` are encoded into these objects.
    b.  A Vector of the linear lower bounds `αlb`
    c.  A Vector of the linear upper bounds `αub`
    d.  A Vector of the data axis
    e.  A Vector of the data

2. `LinSolver`: This is the super type of which I have written two possible subtypes: `UnboundedLinSolver` and `BoundedLinSolver`.  The former uses pivoted QR to solve the linear subsystem, while the latter uses a nonlinear solver to solve the linear subsystem, accepting bounds to the linear parameters.  There's no such thing as a bounded linear solver that I am aware of.  To create either solver, you simply need to pass it a `ModelParams` object.

3. `NLSolver`: This is the super type of which I have written two possible subtypes: `NLGlobalSolver` and `NLLocalSolver`. To create the former you pass it a `LinSolver` object and the maximum running time (a Float64 variable) that you are willing to allow it to run.  The latter is created by passing a `LinSolver` and the relative `xtol` (a Float64 variable) that you want the solver to refine the nonlinear parameter vector to.

Once you've created these 3 objects, you run the `NLSolver` object by simply passing in an initial guess of the non-linear parameters.  My typical use case is to use a very generic guess (like the midpoint of the nonlinear bounds) for the global solver, and then when that finishes I pass it to a local solver for refinement.  It's best to use the `UnboundedLinSolver` for local refinement, as this is a more efficient solver (in both memory and time).  Once you've found a good solution, the error of the fit can be found by bootstrapping.  To do this, I recommend using the `NLLocalSolver` seeded with the best solution of the `NLGlobalSolver` acting on a set of resampled data.

## Planned & Implemented features:

Implemented:

1. Bounded & unbounded linear solvers for the separable linear part of the problem
2. Global and local nonlinear solvers for the nonlinear part of the problem

Todo:

1. Both nonlinear solvers are gradient free at the moment, but when the unbounded linear solver is used there is an analytic expression for the gradient.  We should use that when its possible.
2. Present code requires the data vector to be a vector.  We could easily generalize to "multiple right hand sides", where the target of the fit is an array instead of a vector.  Just needs a bit of code reworking to make that happen.
3.  Bootstrap error calculation
