import numpy as np
from scipy.linalg import eig 


def power_iteration(A, num_iterations=1000, tolerance=1e-6):
    b_k = np.random.rand(A.shape[0])

    for _ in range(num_iterations):
        b_k_next = np.dot(A, b_k)
        b_k_norm = np.linalg.norm(b_k_next)

        b_k_next /= b_k_norm

        if np.linalg.norm(b_k - b_k_next) < tolerance:
            break
        
        b_k = b_k_next 

    eigenvalue = np.dot(b_k.T, np.dot(A, b_k)) / np.dot(b_k.T, b_k)
    return eigenvalue, b_k


def compare_with_library(A):

    own_eigenvalue, own_eigenvector = power_iteration(A)
    print("Own Power Iteration:")
    print("Dominant Eigenvalue:", own_eigenvalue)
    print("Corresponding Eigenvector:", own_eigenvector)

    eigenvalues, eigenvectors = eig(A)

    dominant_index = np.argmax(np.abs(eigenvalues))
    lib_eigenvalue = eigenvalues[dominant_index]
    lib_eigenvector = eigenvectors[:, dominant_index]

    print("\nLibrary Function (SciPy):")
    print("Dominant Eigenvalue:", lib_eigenvalue)
    print("Corresponding Eigenvector:", lib_eigenvector)

    # Comparison
    eigenvalue_diff = np.abs(own_eigenvalue - lib_eigenvalue)
    eigenvector_diff = np.linalg.norm(own_eigenvector - lib_eigenvector)

    print("\nComparison:")
    print(f"Difference in Eigenvalues: {eigenvalue_diff:.6e}")
    print(f"Difference in Eigenvectors (L2 norm): {eigenvector_diff:.6e}")


if __name__ == "__main__":
    A = np.array([[10, 2],
                  [4, 8]])

    compare_with_library(A)
