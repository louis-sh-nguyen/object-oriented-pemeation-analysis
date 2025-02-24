# Technical Documentation: Variable FVT Model Implementation

## 1. Newton's Method Implementation

### Mathematical Framework

The Newton's method is used to solve the non-linear system of equations arising from the Finite Volume Time (FVT) discretization. The iterative formula is:

x_{k+1} = x_k - J(x_k)^{-1} * F(x_k)

where:
- x_k is the solution vector at iteration k
- J(x_k) is the Jacobian matrix evaluated at x_k
- F(x_k) is the residual vector (the system of equations we want to solve, i.e., F(x) = 0)

### `_newton_update_jit` Function

This function performs the core Newton update. It takes the current solution `D_old`, the time step `dt`, spatial step `dx`, parameter `K`, maximum iterations `max_iter`, boundary conditions `D1_prime` and `D2_prime`, relaxation factor `relax`, and relative tolerance `rel_tol` as inputs. It computes the updated solution `D_new`.

### Jacobian Calculation Details

The Jacobian matrix is calculated numerically. The key is to derive the diagonal elements `J_diag`.

Starting from the discretized diffusion equation:

(D_new[i] - D_old[i]) / dt = K * D_new[i] * lapl

where `lapl` is the Laplacian operator:

lapl = (D_new[i+1] - 2.0 * D_new[i] + D_new[i-1]) / dx2

The residual equation is:

R[i] = (D_new[i] - D_old[i]) / dt - K * D_new[i] * lapl

The diagonal element of the Jacobian is the partial derivative of R[i] with respect to D_new[i]:

J_diag[i] = ∂R[i] / ∂D_new[i]

Taking the derivative:

J_diag[i] = (1.0 / dt) - K * (lapl + D_new[i] * (-2.0 / dx2))

Let's break down each term:

*   **(1.0 / dt):** This term comes from the time derivative part of the residual equation, (D_new[i] - D_old[i]) / dt. When we take the partial derivative of this term with respect to D_new[i], we get 1 / dt. This represents the change in the residual due to the change in the concentration at the current time step.

*   **K * (lapl + D_new[i] * (-2.0 / dx2)):** This term comes from the spatial diffusion part of the residual equation, - K * D_new[i] * lapl. Here, K is a constant. Let's analyze the terms inside the parentheses:

    *   **lapl:** This is the Laplacian operator, which represents the spatial second derivative of the concentration. It's defined as (D_new[i+1] - 2.0 * D_new[i] + D_new[i-1]) / dx2. When we take the partial derivative of the term - K * D_new[i] * lapl with respect to D_new[i], we need to apply the product rule. The derivative of D_new[i] is 1, so we get -K * lapl as one part of the derivative.

    *   **D_new[i] \* (-2.0 / dx2):** This term arises from differentiating the Laplacian itself with respect to D_new[i]. The Laplacian contains the term -2.0 \* D_new[i] / dx2. When we differentiate this with respect to D_new[i], we get -2.0 / dx2. This term is then multiplied by D_new[i] and K, resulting in K \* D_new[i] \* (-2.0 / dx2).

Simplifying, we get:

J_diag = (1.0 / dt) - K * (lapl + (-2.0 * D_new[i]) / dx2)

This equation represents the diagonal elements of the Jacobian, taking into account the time discretization, the spatial discretization (Laplacian), and the variable diffusion coefficient.

**Reason for Using Only the Jacobian Diagonal**

In this simplified implementation of Newton's method, only the diagonal elements of the Jacobian are used for computational efficiency. This approach is equivalent to using a Jacobi method or a damped Newton method. The rationale behind this simplification is as follows:

1.  **Computational Cost:** Calculating and storing the full Jacobian matrix can be computationally expensive, especially for large systems. Approximating the Jacobian by its diagonal significantly reduces the memory requirements and computational cost per iteration.

2.  **Diagonal Dominance:** In many diffusion problems, the diagonal elements of the Jacobian are dominant. This means that the influence of a variable on its own residual equation is much stronger than its influence on other residual equations. In such cases, using only the diagonal elements can still provide a reasonable approximation of the Newton update direction.

3.  **Simplified Implementation:** Using only the diagonal elements simplifies the implementation of the Newton update. It avoids the need for solving a linear system of equations at each iteration, which can be computationally expensive.

However, it's important to note that using only the diagonal elements can also have some drawbacks:

1.  **Slower Convergence:** The convergence rate of the simplified Newton method may be slower than that of the full Newton method, especially for problems where the off-diagonal elements of the Jacobian are significant.

2.  **Reduced Accuracy:** The accuracy of the solution may be lower than that of the full Newton method, especially for problems where the off-diagonal elements of the Jacobian are significant.

Despite these drawbacks, using only the diagonal elements of the Jacobian can be a reasonable trade-off between computational cost and accuracy for many diffusion problems, especially when combined with adaptive time stepping and relaxation techniques.

## 2. Variable Adaptive Fitting Schemes

### Time Step Adaptation

The time step `dt` is adapted based on the convergence rate of the Newton's method. If the solution converges quickly (i.e., the residual decreases significantly in each iteration), the time step is increased. If the solution converges slowly or diverges, the time step is decreased.

### Error Control

Error control is implemented by monitoring the norm of the residual vector. The algorithm aims to keep the residual below a specified tolerance `rel_tol`. If the residual exceeds the tolerance, the time step is reduced, and the Newton's method is re-applied.

### Adaptive Relaxation

The relaxation factor `relax` is used to control the step size in the Newton's method. If the solution oscillates or diverges, the relaxation factor is reduced to dampen the oscillations and improve convergence.

## 3. Advanced Tricks to Speed Up Calculation

### JIT Compilation with Numba

The code uses Numba's `@njit` decorator to compile critical functions to machine code. This significantly improves performance, especially for computationally intensive tasks like the Jacobian calculation and the Newton update.

### Vectorization with NumPy

NumPy is used extensively to vectorize operations, avoiding explicit loops in Python. This allows the code to leverage optimized BLAS/LAPACK libraries for linear algebra operations.

### In-Place Operations

The code uses in-place operations where possible to minimize memory allocation and improve performance. For example, the `D_new` array is updated in-place during the Newton iterations.
