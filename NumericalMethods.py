import numpy as np
import matplotlib.pyplot as plt
import time


Num_subdivisions = int(input("Enter the length of the rod ")or 30)
G=float(input("Enter the thermal conductivity factor(0.005041 as an example)")or 0.005041)
T0=int(input("Enter the temperature at the start of the rod in celsius: ")or 100)
T30=int(input("Enter the temperature at the end of the rod in celsius: ")or 100)


main_diag = np.full(Num_subdivisions - 1, -2 - G )  
off_diag = np.ones(Num_subdivisions - 2)
b = np.zeros(Num_subdivisions - 1)
b[0] -= T0
b[-1] -= T30


def thomas_algorithm(off_diag, b, c, d):
    n = len(d)
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - off_diag[i - 1] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - off_diag[i - 1] * d_prime[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


T_inner = thomas_algorithm(off_diag, main_diag, off_diag, b)
T = np.concatenate(([T0], T_inner, [T30]))
x = list(range(0, Num_subdivisions + 1))


def is_diagonally_dominant(A):
    n = len(A)
    for i in range(n):
        sum_row = sum(abs(A[i][j]) for j in range(n) if i != j)
        if abs(A[i][i]) < sum_row:
            return False
    return True
def jacobi(A, b, tolerance, max_iterations=1000):
    n = len(A)
    x = np.zeros_like(b, dtype=np.float64)
    x_new = np.copy(x)
    iterations = 0
    for _ in range(max_iterations):
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - s) / A[i][i]
        iterations += 1
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            return x_new, iterations
        x = np.copy(x_new)
    return x_new, iterations
def gauss_seidel(A, b, tolerance, max_iterations=1000):
    n = len(A)
    x = np.zeros_like(b, dtype=np.float64)
    iterations = 0
    for _ in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        iterations += 1
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            return x_new, iterations
        x = np.copy(x_new)
    return x_new, iterations
def sor(A, b, tolerance, omega=1.25, max_iterations=1000):
    n = len(A)
    x = np.zeros_like(b, dtype=np.float64)
    iterations = 0
    for _ in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            x_new[i] = (1 - omega) * x[i] + (omega * (b[i] - s1 - s2)) / A[i][i]
        iterations += 1
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            return x_new, iterations
        x = np.copy(x_new)
    return x_new, iterations


tolerance = float(input("Enter the tolerance: "))
A = np.zeros((Num_subdivisions-1,Num_subdivisions-1))
B = np.zeros(Num_subdivisions-1)
B[0] = -T0
B[Num_subdivisions-2] = -T30
np.fill_diagonal(A, main_diag)  
np.fill_diagonal(A[1:], off_diag)  
np.fill_diagonal(A[:, 1:], off_diag)

if is_diagonally_dominant(A):
    print("The matrix is diagonally dominant, the methods should converge.")
else:
    print("Warning: The matrix is not diagonally dominant, convergence is not guaranteed.")


start_time = time.time()
jacobi_solution, jacobi_iterations = jacobi(A, B, tolerance)
jacobi_runtime = time.time() - start_time
print(f"\nJacobi method solution: {jacobi_solution}")
print(f"Jacobi method iterations: {jacobi_iterations}")
print(f"Jacobi method runtime: {jacobi_runtime:.5f} seconds")


start_time = time.time()
gs_solution, gs_iterations = gauss_seidel(A, B, tolerance)
gs_runtime = time.time() - start_time
print(f"\nGauss-Seidel method solution: {gs_solution}")
print(f"Gauss-Seidel method iterations: {gs_iterations}")
print(f"Gauss-Seidel method runtime: {gs_runtime:.5f} seconds")


omega = float(input("\nEnter the relaxation factor for SOR (1.5 for diagonal matrix is optimal): ") or 1.5)
start_time = time.time()
sor_solution, sor_iterations = sor(A, B, tolerance, omega)
sor_runtime = time.time() - start_time
print(f"\nSOR method solution: {sor_solution}")
print(f"SOR method iterations: {sor_iterations}")
print(f"SOR method runtime: {sor_runtime:.5f} seconds")


jacobi_temperature = np.concatenate(([T0], jacobi_solution, [T30]))
gs_temperature = np.concatenate(([T0], gs_solution, [T30]))
sor_temperature = np.concatenate(([T0], sor_solution, [T30]))


plt.figure("Temp Distribution")
plt.plot(x, T, marker='o', label='Thomas Algorithm', linestyle='-')
plt.plot(x, jacobi_temperature, marker='x', label='Jacobi Method', linestyle='--')
plt.plot(x, gs_temperature, marker='s', label='Gauss-Seidel Method', linestyle='-.')
plt.plot(x, sor_temperature, marker='d', label=f'SOR Method (ω={omega})', linestyle=':')
plt.title('Temperature Distribution Along the Rod')
plt.xlabel('x (cm)')
plt.ylabel('Temperature  T(x) (°C)')
plt.legend()
plt.show()

runtimevariable=['Jacobi Method', 'Gauss Seidal Method', 'Succesive over Relaxation']
iteravariable=['Jacobi Method', 'Gauss Seidal Method', 'Succesive over Relaxation']
runtimeval=[jacobi_runtime,gs_runtime,sor_runtime]
iteraval=[jacobi_iterations,gs_iterations,sor_iterations]

plt.figure("Run Time graph")  
plt.bar(runtimevariable, runtimeval, color='skyblue')
plt.title("Run Time graph")
plt.xlabel("Methods for solving linear system of equations")
plt.ylabel("Run Time(in seconds)")
plt.show()

plt.figure("iterationswaa")  
plt.bar(iteravariable, iteraval, color='orange')
plt.title("Iteration count graph")
plt.xlabel("Methods for solving linear system of equations")
plt.ylabel("Number of iterations")
plt.show()

