"""
Tests for Linear Algebra module.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from math_utils.linear_algebra import LinearAlgebra


class TestBasicOperations:
    """Test basic linear algebra operations."""
    
    def test_dot_product(self):
        """Test dot product computation."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        result = LinearAlgebra.dot_product(v1, v2)
        expected = 32  # 1*4 + 2*5 + 3*6
        assert np.isclose(result, expected)
    
    def test_dot_product_orthogonal(self):
        """Test dot product of orthogonal vectors."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        result = LinearAlgebra.dot_product(v1, v2)
        assert np.isclose(result, 0.0)
    
    def test_cross_product(self):
        """Test cross product."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        result = LinearAlgebra.cross_product(v1, v2)
        expected = np.array([0, 0, 1])
        assert np.allclose(result, expected)
    
    def test_matrix_multiply(self):
        """Test matrix multiplication."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        result = LinearAlgebra.matrix_multiply(A, B)
        expected = np.array([[19, 22], [43, 50]])
        assert np.allclose(result, expected)
    
    def test_outer_product(self):
        """Test outer product."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5])
        result = LinearAlgebra.outer_product(v1, v2)
        expected = np.array([[4, 5], [8, 10], [12, 15]])
        assert np.allclose(result, expected)


class TestMatrixProperties:
    """Test matrix property computations."""
    
    def test_determinant(self):
        """Test determinant computation."""
        A = np.array([[1, 2], [3, 4]])
        result = LinearAlgebra.determinant(A)
        expected = -2
        assert np.isclose(result, expected)
    
    def test_determinant_identity(self):
        """Test determinant of identity matrix."""
        A = np.eye(5)
        result = LinearAlgebra.determinant(A)
        assert np.isclose(result, 1.0)
    
    def test_trace(self):
        """Test trace computation."""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = LinearAlgebra.trace(A)
        expected = 15  # 1 + 5 + 9
        assert np.isclose(result, expected)
    
    def test_rank(self):
        """Test rank computation."""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = LinearAlgebra.rank(A)
        assert result == 2  # This matrix has rank 2
    
    def test_rank_full(self):
        """Test rank of full rank matrix."""
        A = np.eye(5)
        result = LinearAlgebra.rank(A)
        assert result == 5
    
    def test_is_symmetric(self):
        """Test symmetric matrix check."""
        A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        assert LinearAlgebra.is_symmetric(A)
        
        B = np.array([[1, 2], [3, 4]])
        assert not LinearAlgebra.is_symmetric(B)
    
    def test_is_positive_definite(self):
        """Test positive definite check."""
        A = np.array([[2, 0], [0, 3]])
        assert LinearAlgebra.is_positive_definite(A)
        
        B = np.array([[-1, 0], [0, -2]])
        assert not LinearAlgebra.is_positive_definite(B)
    
    def test_is_orthogonal(self):
        """Test orthogonal matrix check."""
        # Rotation matrix
        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
        assert LinearAlgebra.is_orthogonal(R)


class TestDecompositions:
    """Test matrix decompositions."""
    
    def test_lu_decomposition(self):
        """Test LU decomposition."""
        A = np.array([[4, 3], [6, 3]])
        L, U, P = LinearAlgebra.lu_decomposition(A)
        
        # Check that PA = LU
        assert np.allclose(P @ A, L @ U)
        
        # Check L is lower triangular
        assert np.allclose(L, np.tril(L))
        
        # Check U is upper triangular
        assert np.allclose(U, np.triu(U))
    
    def test_qr_decomposition(self):
        """Test QR decomposition."""
        A = np.array([[1, 2], [3, 4], [5, 6]])
        Q, R = LinearAlgebra.qr_decomposition(A)
        
        # Check that A = QR
        assert np.allclose(A, Q @ R)
        
        # Check Q is orthogonal
        assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)
        
        # Check R is upper triangular
        assert np.allclose(R, np.triu(R))
    
    def test_cholesky_decomposition(self):
        """Test Cholesky decomposition."""
        A = np.array([[4, 2], [2, 5]])
        L = LinearAlgebra.cholesky_decomposition(A)
        
        # Check that A = LL^T
        assert np.allclose(A, L @ L.T)
        
        # Check L is lower triangular
        assert np.allclose(L, np.tril(L))
    
    def test_svd(self):
        """Test SVD decomposition."""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        U, S, Vt = LinearAlgebra.svd(A)
        
        # Reconstruct A
        Sigma = np.zeros_like(A, dtype=float)
        np.fill_diagonal(Sigma, S)
        A_reconstructed = U @ Sigma @ Vt
        
        assert np.allclose(A_reconstructed, A)
    
    def test_eigendecomposition(self):
        """Test eigendecomposition."""
        A = np.array([[4, 1], [2, 3]])
        eigenvalues, eigenvectors = LinearAlgebra.eigendecomposition(A)
        
        # Check that Av = λv for each eigenpair
        for i in range(len(eigenvalues)):
            v = eigenvectors[:, i]
            λ = eigenvalues[i]
            assert np.allclose(A @ v, λ * v)


class TestLinearSystems:
    """Test linear system solving."""
    
    def test_solve(self):
        """Test solving linear systems."""
        A = np.array([[3, 1], [1, 2]])
        b = np.array([9, 8])
        x = LinearAlgebra.solve(A, b)
        
        # Check that Ax = b
        assert np.allclose(A @ x, b)
    
    def test_least_squares(self):
        """Test least squares solution."""
        # Overdetermined system
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([2, 3, 5])
        x = LinearAlgebra.least_squares(A, b)
        
        # Check that x minimizes ||Ax - b||^2
        residual = A @ x - b
        assert np.allclose(A.T @ residual, 0, atol=1e-10)
    
    def test_pseudoinverse(self):
        """Test Moore-Penrose pseudoinverse."""
        A = np.array([[1, 2], [3, 4], [5, 6]])
        A_pinv = LinearAlgebra.pseudoinverse(A)
        
        # Check Penrose conditions
        assert np.allclose(A @ A_pinv @ A, A)
        assert np.allclose(A_pinv @ A @ A_pinv, A_pinv)


class TestNormsAndDistances:
    """Test norms and distance computations."""
    
    def test_norm_l1(self):
        """Test L1 norm."""
        v = np.array([3, 4])
        result = LinearAlgebra.norm(v, ord=1)
        assert np.isclose(result, 7)
    
    def test_norm_l2(self):
        """Test L2 norm."""
        v = np.array([3, 4])
        result = LinearAlgebra.norm(v, ord=2)
        assert np.isclose(result, 5)
    
    def test_norm_inf(self):
        """Test infinity norm."""
        v = np.array([3, -7, 2])
        result = LinearAlgebra.norm(v, ord=np.inf)
        assert np.isclose(result, 7)
    
    def test_euclidean_distance(self):
        """Test Euclidean distance."""
        v1 = np.array([0, 0])
        v2 = np.array([3, 4])
        result = LinearAlgebra.euclidean_distance(v1, v2)
        assert np.isclose(result, 5)
    
    def test_cosine_similarity(self):
        """Test cosine similarity."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        result = LinearAlgebra.cosine_similarity(v1, v2)
        assert np.isclose(result, 0)
        
        # Same vector should have similarity 1
        result = LinearAlgebra.cosine_similarity(v1, v1)
        assert np.isclose(result, 1)
    
    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite vectors."""
        v1 = np.array([1, 0, 0])
        v2 = np.array([-1, 0, 0])
        result = LinearAlgebra.cosine_similarity(v1, v2)
        assert np.isclose(result, -1)


class TestVectorSpace:
    """Test vector space operations."""
    
    def test_gram_schmidt(self):
        """Test Gram-Schmidt orthonormalization."""
        vectors = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]).T
        Q = LinearAlgebra.gram_schmidt(vectors)
        
        # Check orthogonality
        for i in range(Q.shape[1]):
            for j in range(i + 1, Q.shape[1]):
                assert np.isclose(np.dot(Q[:, i], Q[:, j]), 0, atol=1e-10)
        
        # Check normalization
        for i in range(Q.shape[1]):
            assert np.isclose(np.linalg.norm(Q[:, i]), 1)
    
    def test_projection(self):
        """Test vector projection onto subspace."""
        v = np.array([3, 4, 5])
        subspace = np.array([[1, 0, 0], [0, 1, 0]]).T
        
        proj = LinearAlgebra.projection(v, subspace)
        
        # Projection onto xy-plane should zero out z component
        assert np.isclose(proj[2], 0)
        assert np.isclose(proj[0], v[0])
        assert np.isclose(proj[1], v[1])


class TestAdvancedOperations:
    """Test advanced operations."""
    
    def test_matrix_exponential(self):
        """Test matrix exponential."""
        A = np.array([[1, 0], [0, 2]])
        exp_A = LinearAlgebra.matrix_exponential(A)
        
        # For diagonal matrix, exp(A) = diag(exp(λ_i))
        expected = np.diag([np.exp(1), np.exp(2)])
        assert np.allclose(exp_A, expected)
    
    def test_hadamard_product(self):
        """Test Hadamard (element-wise) product."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        result = LinearAlgebra.hadamard_product(A, B)
        expected = np.array([[5, 12], [21, 32]])
        assert np.allclose(result, expected)
    
    def test_condition_number(self):
        """Test condition number computation."""
        A = np.eye(5)
        cond = LinearAlgebra.condition_number(A)
        assert np.isclose(cond, 1.0)
        
        # Ill-conditioned matrix
        B = np.array([[1, 0], [0, 1e-10]])
        cond = LinearAlgebra.condition_number(B)
        assert cond > 1e9
    
    def test_householder_reflection(self):
        """Test Householder reflection."""
        v = np.array([1, 1, 1])
        H = LinearAlgebra.householder_reflection(v)
        
        # H should be symmetric
        assert np.allclose(H, H.T)
        
        # H should be orthogonal
        assert np.allclose(H @ H.T, np.eye(3))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
