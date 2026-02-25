"""
Linear Algebra Module

Comprehensive linear algebra operations for machine learning and AI applications.
Includes matrix operations, decompositions, and vector space computations.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from numpy.linalg import LinAlgError


class LinearAlgebra:
    """
    A comprehensive class for linear algebra operations.
    
    Methods cover:
    - Basic matrix/vector operations
    - Matrix decompositions (LU, QR, Cholesky, SVD, Eigendecomposition)
    - Vector space operations
    - Norms and distances
    - Solving linear systems
    """
    
    # ==================== Basic Operations ====================
    
    @staticmethod
    def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute the dot product of two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Dot product scalar
            
        Raises:
            ValueError: If vectors have different shapes
        """
        if v1.shape != v2.shape:
            raise ValueError(f"Vectors must have same shape. Got {v1.shape} and {v2.shape}")
        return np.sum(v1 * v2)
    
    @staticmethod
    def cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Compute the cross product of two 3D vectors.
        
        Args:
            v1: First 3D vector
            v2: Second 3D vector
            
        Returns:
            Cross product vector
            
        Raises:
            ValueError: If vectors are not 3D
        """
        if v1.shape != (3,) or v2.shape != (3,):
            raise ValueError("Cross product is only defined for 3D vectors")
        return np.cross(v1, v2)
    
    @staticmethod
    def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Perform matrix multiplication using Strassen's algorithm for large matrices.
        
        Args:
            A: First matrix (m x n)
            B: Second matrix (n x p)
            
        Returns:
            Product matrix (m x p)
            
        Raises:
            ValueError: If matrices have incompatible shapes
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Incompatible shapes: {A.shape} and {B.shape}")
        
        # Use numpy's optimized implementation
        return np.matmul(A, B)
    
    @staticmethod
    def outer_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Compute the outer product of two vectors.
        
        Args:
            v1: First vector (m,)
            v2: Second vector (n,)
            
        Returns:
            Outer product matrix (m x n)
        """
        return np.outer(v1, v2)
    
    @staticmethod
    def kronecker_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute the Kronecker product of two matrices.
        
        Args:
            A: First matrix
            B: Second matrix
            
        Returns:
            Kronecker product matrix
        """
        return np.kron(A, B)
    
    # ==================== Matrix Properties ====================
    
    @staticmethod
    def determinant(A: np.ndarray) -> float:
        """
        Compute the determinant of a square matrix.
        
        Args:
            A: Square matrix
            
        Returns:
            Determinant value
            
        Raises:
            ValueError: If matrix is not square
        """
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Determinant requires a square matrix")
        return np.linalg.det(A)
    
    @staticmethod
    def trace(A: np.ndarray) -> float:
        """
        Compute the trace of a square matrix.
        
        Args:
            A: Square matrix
            
        Returns:
            Trace (sum of diagonal elements)
        """
        return np.trace(A)
    
    @staticmethod
    def rank(A: np.ndarray, tol: float = 1e-10) -> int:
        """
        Compute the rank of a matrix.
        
        Args:
            A: Input matrix
            tol: Tolerance for singular values
            
        Returns:
            Matrix rank
        """
        _, s, _ = np.linalg.svd(A)
        return np.sum(s > tol)
    
    @staticmethod
    def is_symmetric(A: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if a matrix is symmetric."""
        return A.ndim == 2 and A.shape[0] == A.shape[1] and np.allclose(A, A.T, atol=tol)
    
    @staticmethod
    def is_positive_definite(A: np.ndarray) -> bool:
        """
        Check if a matrix is positive definite.
        
        Args:
            A: Square symmetric matrix
            
        Returns:
            True if positive definite
        """
        if not LinearAlgebra.is_symmetric(A):
            return False
        try:
            np.linalg.cholesky(A)
            return True
        except LinAlgError:
            return False
    
    @staticmethod
    def is_orthogonal(A: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if a matrix is orthogonal."""
        return np.allclose(A @ A.T, np.eye(A.shape[0]), atol=tol)
    
    # ==================== Matrix Decompositions ====================
    
    @staticmethod
    def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform LU decomposition with partial pivoting.
        
        Args:
            A: Square matrix
            
        Returns:
            Tuple of (L, U, P) where PA = LU
            L: Lower triangular matrix
            U: Upper triangular matrix
            P: Permutation matrix
            
        Raises:
            ValueError: If matrix is not square
        """
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("LU decomposition requires a square matrix")
        
        n = A.shape[0]
        L = np.eye(n)
        U = A.copy().astype(float)
        P = np.eye(n)
        
        for k in range(n - 1):
            # Find pivot
            max_idx = k + np.argmax(np.abs(U[k:, k]))
            
            if max_idx != k:
                # Swap rows
                U[[k, max_idx]] = U[[max_idx, k]]
                P[[k, max_idx]] = P[[max_idx, k]]
                if k > 0:
                    L[[k, max_idx], :k] = L[[max_idx, k], :k]
            
            # Elimination
            for i in range(k + 1, n):
                if U[k, k] != 0:
                    L[i, k] = U[i, k] / U[k, k]
                    U[i, k:] -= L[i, k] * U[k, k:]
        
        return L, U, P
    
    @staticmethod
    def qr_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform QR decomposition using Gram-Schmidt process.
        
        Args:
            A: Matrix (m x n)
            
        Returns:
            Tuple of (Q, R) where A = QR
            Q: Orthogonal matrix (m x m)
            R: Upper triangular matrix (m x n)
        """
        m, n = A.shape
        Q = np.zeros((m, n))
        R = np.zeros((n, n))
        
        for j in range(n):
            v = A[:, j].copy()
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], A[:, j])
                v -= R[i, j] * Q[:, i]
            R[j, j] = np.linalg.norm(v)
            if R[j, j] > 1e-10:
                Q[:, j] = v / R[j, j]
            else:
                Q[:, j] = v
        
        return Q, R
    
    @staticmethod
    def cholesky_decomposition(A: np.ndarray) -> np.ndarray:
        """
        Perform Cholesky decomposition for positive definite matrices.
        
        Args:
            A: Positive definite symmetric matrix
            
        Returns:
            Lower triangular matrix L where A = LL^T
            
        Raises:
            LinAlgError: If matrix is not positive definite
        """
        return np.linalg.cholesky(A)
    
    @staticmethod
    def svd(A: np.ndarray, full_matrices: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Singular Value Decomposition.
        
        Args:
            A: Input matrix (m x n)
            full_matrices: If True, return full U and V^T matrices
            
        Returns:
            Tuple of (U, S, Vt) where A = U @ diag(S) @ Vt
            U: Left singular vectors (m x m) or (m x k)
            S: Singular values (k,) where k = min(m, n)
            Vt: Right singular vectors transposed (n x n) or (k x n)
        """
        U, S, Vt = np.linalg.svd(A, full_matrices=full_matrices)
        return U, S, Vt
    
    @staticmethod
    def eigendecomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform eigendecomposition of a square matrix.
        
        Args:
            A: Square matrix
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
            eigenvalues: Array of eigenvalues
            eigenvectors: Matrix where columns are eigenvectors
            
        Raises:
            ValueError: If matrix is not square
        """
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Eigendecomposition requires a square matrix")
        
        eigenvalues, eigenvectors = np.linalg.eig(A)
        return eigenvalues, eigenvectors
    
    @staticmethod
    def schur_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Schur decomposition.
        
        Args:
            A: Square matrix
            
        Returns:
            Tuple of (T, U) where A = U @ T @ U^H
            T: Upper triangular matrix
            U: Unitary matrix
        """
        T, U = np.linalg.schur(A)
        return T, U
    
    # ==================== Solving Linear Systems ====================
    
    @staticmethod
    def solve(A: np.ndarray, b: np.ndarray, method: str = "lu") -> np.ndarray:
        """
        Solve a system of linear equations Ax = b.
        
        Args:
            A: Coefficient matrix
            b: Right-hand side vector
            method: Solution method ('lu', 'qr', 'cholesky', 'svd')
            
        Returns:
            Solution vector x
            
        Raises:
            ValueError: If system is not solvable
        """
        if method == "lu":
            L, U, P = LinearAlgebra.lu_decomposition(A)
            # Solve Ly = Pb
            y = np.linalg.solve(L, P @ b)
            # Solve Ux = y
            x = np.linalg.solve(U, y)
            return x
        elif method == "qr":
            Q, R = LinearAlgebra.qr_decomposition(A)
            return np.linalg.solve(R, Q.T @ b)
        elif method == "cholesky":
            L = LinearAlgebra.cholesky_decomposition(A)
            y = np.linalg.solve(L, b)
            return np.linalg.solve(L.T, y)
        else:
            return np.linalg.solve(A, b)
    
    @staticmethod
    def least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve the least squares problem min ||Ax - b||^2.
        
        Args:
            A: Design matrix (m x n)
            b: Observation vector (m,)
            
        Returns:
            Least squares solution (n,)
        """
        # Normal equations: A^T A x = A^T b
        return np.linalg.lstsq(A, b, rcond=None)[0]
    
    @staticmethod
    def pseudoinverse(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        """
        Compute the Moore-Penrose pseudoinverse.
        
        Args:
            A: Input matrix
            tol: Tolerance for singular values
            
        Returns:
            Pseudoinverse matrix
        """
        U, S, Vt = np.linalg.svd(A)
        # Invert non-zero singular values
        S_inv = np.zeros_like(S)
        mask = S > tol
        S_inv[mask] = 1.0 / S[mask]
        return Vt.T @ np.diag(S_inv) @ U.T
    
    # ==================== Norms and Distances ====================
    
    @staticmethod
    def norm(v: Union[np.ndarray, List], ord: int = 2) -> float:
        """
        Compute vector norm.
        
        Args:
            v: Input vector
            ord: Order of norm (1, 2, inf, etc.)
            
        Returns:
            Norm value
        """
        return np.linalg.norm(v, ord=ord)
    
    @staticmethod
    def matrix_norm(A: np.ndarray, ord: str = "fro") -> float:
        """
        Compute matrix norm.
        
        Args:
            A: Input matrix
            ord: Norm type ('fro', 1, 2, inf, 'nuc')
            
        Returns:
            Norm value
        """
        return np.linalg.norm(A, ord=ord)
    
    @staticmethod
    def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors."""
        return np.linalg.norm(v1 - v2)
    
    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Cosine similarity in range [-1, 1]
        """
        dot = np.dot(v1.flatten(), v2.flatten())
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 0.0
        return dot / norm_product
    
    @staticmethod
    def mahalanobis_distance(v1: np.ndarray, v2: np.ndarray, 
                             covariance: np.ndarray) -> float:
        """
        Compute Mahalanobis distance between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            covariance: Covariance matrix
            
        Returns:
            Mahalanobis distance
        """
        diff = v1 - v2
        cov_inv = np.linalg.inv(covariance)
        return np.sqrt(diff.T @ cov_inv @ diff)
    
    # ==================== Vector Space Operations ====================
    
    @staticmethod
    def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
        """
        Orthonormalize vectors using Gram-Schmidt process.
        
        Args:
            vectors: Matrix where columns are vectors to orthonormalize
            
        Returns:
            Orthonormalized vectors (same shape)
        """
        m, n = vectors.shape
        Q = np.zeros((m, n))
        
        for j in range(n):
            v = vectors[:, j].copy()
            for i in range(j):
                v -= np.dot(Q[:, i], vectors[:, j]) * Q[:, i]
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                Q[:, j] = v / norm
        
        return Q
    
    @staticmethod
    def null_space(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        """
        Compute an orthonormal basis for the null space of A.
        
        Args:
            A: Input matrix
            tol: Tolerance for singular values
            
        Returns:
            Matrix whose columns form a basis for the null space
        """
        U, S, Vt = np.linalg.svd(A)
        null_mask = S <= tol
        null_space = Vt[null_mask, :].T
        return null_space
    
    @staticmethod
    def column_space(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        """
        Compute an orthonormal basis for the column space of A.
        
        Args:
            A: Input matrix
            tol: Tolerance for singular values
            
        Returns:
            Matrix whose columns form a basis for the column space
        """
        U, S, Vt = np.linalg.svd(A)
        rank_mask = S > tol
        return U[:, rank_mask]
    
    @staticmethod
    def projection(v: np.ndarray, subspace: np.ndarray) -> np.ndarray:
        """
        Project a vector onto a subspace.
        
        Args:
            v: Vector to project
            subspace: Matrix whose columns form a basis for the subspace
            
        Returns:
            Projection of v onto the subspace
        """
        # P = A(A^T A)^(-1) A^T
        A = subspace
        P = A @ np.linalg.inv(A.T @ A) @ A.T
        return P @ v
    
    # ==================== Advanced Operations ====================
    
    @staticmethod
    def matrix_exponential(A: np.ndarray) -> np.ndarray:
        """Compute the matrix exponential e^A."""
        from scipy.linalg import expm
        return expm(A)
    
    @staticmethod
    def matrix_power(A: np.ndarray, n: int) -> np.ndarray:
        """Compute A^n efficiently using eigendecomposition."""
        if n == 0:
            return np.eye(A.shape[0])
        if n < 0:
            A = np.linalg.inv(A)
            n = -n
        
        eigenvalues, eigenvectors = LinearAlgebra.eigendecomposition(A)
        return eigenvectors @ np.diag(eigenvalues ** n) @ np.linalg.inv(eigenvectors)
    
    @staticmethod
    def hadamard_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute element-wise (Hadamard) product of two matrices."""
        if A.shape != B.shape:
            raise ValueError("Matrices must have the same shape")
        return A * B
    
    @staticmethod
    def tensor_product(*arrays: np.ndarray) -> np.ndarray:
        """
        Compute the tensor product of multiple arrays.
        
        Args:
            *arrays: Variable number of arrays
            
        Returns:
            Tensor product array
        """
        result = arrays[0]
        for arr in arrays[1:]:
            result = np.tensordot(result, arr, axes=0)
        return result
    
    @staticmethod
    def batched_dot_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute batched dot product for 3D arrays.
        
        Args:
            A: Array of shape (batch, m, n)
            B: Array of shape (batch, n, p)
            
        Returns:
            Array of shape (batch, m, p)
        """
        return np.matmul(A, B)
    
    @staticmethod
    def condition_number(A: np.ndarray, p: int = 2) -> float:
        """
        Compute the condition number of a matrix.
        
        Args:
            A: Input matrix
            p: Order of norm
            
        Returns:
            Condition number
        """
        return np.linalg.cond(A, p=p)
    
    @staticmethod
    def householder_reflection(v: np.ndarray) -> np.ndarray:
        """
        Compute Householder reflection matrix.
        
        Args:
            v: Vector to reflect
            
        Returns:
            Householder matrix H = I - 2vv^T / v^T v
        """
        v = v.reshape(-1, 1)
        return np.eye(len(v)) - 2 * (v @ v.T) / (v.T @ v)
    
    @staticmethod
    def givens_rotation(n: int, i: int, j: int, theta: float) -> np.ndarray:
        """
        Create a Givens rotation matrix.
        
        Args:
            n: Size of the matrix
            i: First index
            j: Second index
            theta: Rotation angle
            
        Returns:
            Givens rotation matrix
        """
        G = np.eye(n)
        c, s = np.cos(theta), np.sin(theta)
        G[i, i] = c
        G[j, j] = c
        G[i, j] = -s
        G[j, i] = s
        return G
