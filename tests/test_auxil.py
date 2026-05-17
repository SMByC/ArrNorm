"""Unit tests for core.auxil.auxil math primitives (Cpm, geneiv, orthoregress)."""
import numpy as np
import pytest

from core.auxil.auxil import Cpm, geneiv, orthoregress


# ---------------------------------------------------------------------------
# orthoregress
# ---------------------------------------------------------------------------

class TestOrthoregress:

    def test_known_slope_and_intercept(self):
        """Noisy y = 2x + 3 should recover slope≈2, intercept≈3, R≈1."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 100, 1000)
        y = 2.0 * x + 3.0 + rng.normal(0, 0.1, 1000)
        b, a, R = orthoregress(x, y)
        assert abs(b - 2.0) < 0.01
        assert abs(a - 3.0) < 0.2
        assert R > 0.999

    def test_perfect_line_exact_recovery(self):
        """Exactly y = 3x + 5 → R=1, slope=3, intercept=5."""
        x = np.arange(100.0)
        y = 3.0 * x + 5.0
        b, a, R = orthoregress(x, y)
        assert abs(R - 1.0) < 1e-10
        assert abs(b - 3.0) < 1e-8
        assert abs(a - 5.0) < 1e-6

    def test_r_is_symmetric(self):
        """Pearson R must be the same regardless of which variable is x or y."""
        rng = np.random.default_rng(0)
        x = rng.normal(100, 20, 500)
        y = 1.5 * x + 10 + rng.normal(0, 5, 500)
        _, _, R_xy = orthoregress(x, y)
        _, _, R_yx = orthoregress(y, x)
        assert abs(R_xy - R_yx) < 1e-12

    def test_returns_three_values(self):
        result = orthoregress(np.array([1.0, 2.0, 3.0]),
                              np.array([2.0, 4.0, 6.0]))
        assert len(result) == 3

    def test_zero_covariance_gives_zero_slope(self):
        """Constant y → slope=0, intercept=mean(y), R=0."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([5.0, 5.0, 5.0, 5.0])
        b, a, R = orthoregress(x, y)
        assert b == pytest.approx(0.0)
        assert a == pytest.approx(5.0)
        assert R == pytest.approx(0.0)

    def test_negative_slope(self):
        """y = -2x + 10 → slope≈-2."""
        x = np.linspace(0, 50, 500)
        y = -2.0 * x + 10.0
        b, a, R = orthoregress(x, y)
        assert abs(b - (-2.0)) < 1e-8
        assert abs(a - 10.0)  < 1e-5


# ---------------------------------------------------------------------------
# Cpm — weighted streaming covariance accumulator
# ---------------------------------------------------------------------------

class TestCpm:

    def test_covariance_matches_numpy_batch(self):
        """Single batch update must match np.cov to ~1e-8 relative tolerance."""
        rng = np.random.default_rng(123)
        data = rng.normal(0, 1, (500, 4))   # (n_pixels, n_bands)
        cpm = Cpm(4)
        cpm.update(data)
        np.testing.assert_allclose(cpm.covariance(), np.cov(data.T), rtol=1e-8)

    def test_incremental_equals_batch(self):
        """Feeding one row at a time must produce the same result as one batch."""
        rng = np.random.default_rng(55)
        data = rng.normal(0, 1, (100, 3))

        cpm_batch = Cpm(3)
        cpm_batch.update(data)

        cpm_inc = Cpm(3)
        for row in data:
            cpm_inc.update(row)   # 1-D input, reshaped inside update()

        np.testing.assert_allclose(
            cpm_batch.covariance(), cpm_inc.covariance(), rtol=1e-10)

    def test_means_match_numpy(self):
        rng = np.random.default_rng(9)
        data = rng.normal(10, 3, (200, 3))
        cpm = Cpm(3)
        cpm.update(data)
        # West's algorithm carries a 1e-7 seed weight that causes ~5e-10
        # relative error vs the plain sample mean; rtol=1e-8 is sufficient.
        np.testing.assert_allclose(cpm.means(), data.mean(axis=0), rtol=1e-8)

    def test_zero_weight_pixels_have_no_effect(self):
        """Pixels with weight=0 must not change the covariance estimate."""
        rng = np.random.default_rng(7)
        data     = rng.normal(0, 1, (100, 2))
        outliers = rng.normal(0, 100, (10, 2))

        cpm_base = Cpm(2)
        cpm_base.update(data)

        cpm_zero = Cpm(2)
        cpm_zero.update(data)
        cpm_zero.update(outliers, np.zeros(10))

        np.testing.assert_allclose(cpm_base.cov, cpm_zero.cov, rtol=1e-12)

    def test_reset_restores_initial_state(self):
        rng = np.random.default_rng(0)
        cpm = Cpm(3)
        cpm.update(rng.normal(0, 1, (50, 3)))
        cpm.reset()
        assert np.all(cpm.cov == 0)
        assert np.all(cpm.mn  == 0)
        assert cpm.sw == pytest.approx(1e-7)

    def test_reset_then_reuse_gives_same_result(self):
        """After reset, a second pass must reproduce the first pass exactly."""
        rng = np.random.default_rng(11)
        data = rng.normal(5, 2, (80, 2))

        cpm = Cpm(2)
        cpm.update(data)
        cov_first = cpm.covariance().copy()

        cpm.reset()
        cpm.update(data)
        cov_second = cpm.covariance()

        np.testing.assert_array_equal(cov_first, cov_second)

    def test_weighted_covariance_down_weights_outliers(self):
        """High-weight clean data should dominate a near-zero-weight noisy block.

        With weight w=1e-10 and 50 outliers of std=1000, the outlier
        contribution to the scatter matrix is ≈ 1e-10 × 50 × 10⁶ = 5e-3,
        which is ~1e-5 relative to the clean scatter (~500). The covariances
        therefore agree to rtol=1e-3.
        """
        rng = np.random.default_rng(99)
        clean = rng.normal(0, 1, (500, 2))
        noisy = rng.normal(0, 1000, (50, 2))

        cpm_clean = Cpm(2)
        cpm_clean.update(clean, np.ones(500))

        cpm_mixed = Cpm(2)
        cpm_mixed.update(clean, np.ones(500))
        cpm_mixed.update(noisy, np.full(50, 1e-10))

        np.testing.assert_allclose(
            cpm_clean.covariance(), cpm_mixed.covariance(), rtol=1e-3)


# ---------------------------------------------------------------------------
# geneiv — symmetric generalized eigenproblem
# ---------------------------------------------------------------------------

class TestGeneiv:

    def test_standard_eigenproblem(self):
        """geneiv(A, I) eigenvalues must match numpy.linalg.eigvalsh(A)."""
        A = np.array([[4.0, 2.0], [2.0, 3.0]])
        B = np.eye(2)
        vals, _ = geneiv(A, B)
        expected = np.sort(np.linalg.eigvalsh(A))
        np.testing.assert_allclose(np.sort(vals), expected, rtol=1e-8)

    def test_eigenpair_equation_holds(self):
        """A·v = λ·B·v must hold for every returned (λ, v) pair."""
        rng = np.random.default_rng(42)
        M = rng.normal(0, 1, (4, 4))
        A = M.T @ M + np.eye(4)   # symmetric positive-definite
        N = rng.normal(0, 1, (4, 4))
        B = N.T @ N + np.eye(4)
        vals, vecs = geneiv(A, B)
        for i in range(4):
            lhs = A @ vecs[:, i]
            rhs = vals[i] * (B @ vecs[:, i])
            np.testing.assert_allclose(lhs, rhs, rtol=1e-8, atol=1e-10)

    def test_eigenvalues_ascending(self):
        """geneiv must return eigenvalues sorted in ascending order."""
        rng = np.random.default_rng(3)
        M = rng.normal(0, 1, (5, 5))
        A = M.T @ M + np.eye(5)
        N = rng.normal(0, 1, (5, 5))
        B = N.T @ N + np.eye(5)
        vals, _ = geneiv(A, B)
        assert list(vals) == sorted(vals)

    def test_scalar_case(self):
        """1×1 matrices: eigenvalue = A[0,0] / B[0,0]."""
        A = np.array([[6.0]])
        B = np.array([[2.0]])
        vals, _ = geneiv(A, B)
        assert vals[0] == pytest.approx(3.0)

    def test_returns_correct_shapes(self):
        rng = np.random.default_rng(5)
        n = 6
        M = rng.normal(0, 1, (n, n))
        A = M.T @ M + np.eye(n)
        B = np.eye(n)
        vals, vecs = geneiv(A, B)
        assert vals.shape == (n,)
        assert vecs.shape == (n, n)
