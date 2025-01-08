import numpy as np

class SolidElasticity:

    @staticmethod
    def saint_venant_constraint(predicted_positions, reference_positions, p1, p2, p3, p4, youngs_modulus, poisson_ratio, lagrange, dt, masses, fixed_nodes=None, bc_nodes=None):
        """Enforce Saint-Venant elasticity constraint for a tetrahedron."""
        if fixed_nodes is None:
            fixed_nodes = set()

        # Material properties
        mu = youngs_modulus / (2 * (1 + poisson_ratio))
        lambda_ = youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

        # Reference and predicted deformation matrices
        Dm = np.column_stack((
            reference_positions[p1] - reference_positions[p4],
            reference_positions[p2] - reference_positions[p4],
            reference_positions[p3] - reference_positions[p4]
        ))
        Ds = np.column_stack((
            predicted_positions[p1] - predicted_positions[p4],
            predicted_positions[p2] - predicted_positions[p4],
            predicted_positions[p3] - predicted_positions[p4]
        ))

        Dm_inv = np.linalg.inv(Dm)
        F = Ds @ Dm_inv

        I = np.identity(3)
        epsilon = 1e-20

        # Singular Value Decomposition (SVD)
        U, singular_values, Vt = np.linalg.svd(F, full_matrices=True)
        Fhat = np.diag(singular_values)

        if np.linalg.det(F) < 0:
            Fhat[2, 2] = -Fhat[2, 2]
            U[:, 2] = -U[:, 2]

        # Apply compression limit
        min_singular_value = 0.577
        Fhat[0, 0] = max(Fhat[0, 0], min_singular_value)
        Fhat[1, 1] = max(Fhat[1, 1], min_singular_value)
        Fhat[2, 2] = max(Fhat[2, 2], min_singular_value)

        # Green strain tensor and stress computation
        Ehat = 0.5 * (Fhat.T @ Fhat - I)
        Ehat_trace = np.trace(Ehat)
        Piolahat = Fhat @ (2 * mu * Ehat + lambda_ * Ehat_trace * I)

        E = U @ Ehat @ Vt
        Etrace = np.trace(E)
        psi = mu * np.sum(E**2) + 0.5 * lambda_ * Etrace**2

        Piola = U @ Piolahat @ Vt

        V0 = abs(np.linalg.det(Dm)) / 6.0
        H = -V0 * Piola @ Dm_inv.T

        f1 = H[:, 0]
        f2 = H[:, 1]
        f3 = H[:, 2]
        f4 = -(f1 + f2 + f3)

        w1 = 1.0 / masses[p1]
        w2 = 1.0 / masses[p2]
        w3 = 1.0 / masses[p3]
        w4 = 1.0 / masses[p4]

        weighted_sum_of_gradients = (
            w1 * np.dot(f1, f1) +
            w2 * np.dot(f2, f2) +
            w3 * np.dot(f3, f3) +
            w4 * np.dot(f4, f4)
        )

        if weighted_sum_of_gradients < epsilon:
            return

        C = V0 * psi
        alpha_tilde = 1.0 / (dt**2)
        delta_lagrange = -(C + alpha_tilde * lagrange) / (weighted_sum_of_gradients + alpha_tilde)

        lagrange += delta_lagrange        
        
        # Update predicted positions unless they are fixed
        if p1 not in bc_nodes:
            predicted_positions[p1] += w1 * -f1 * delta_lagrange
        if p2 not in bc_nodes:
            predicted_positions[p2] += w2 * -f2 * delta_lagrange
        if p3 not in bc_nodes:
            predicted_positions[p3] += w3 * -f3 * delta_lagrange
        if p4 not in bc_nodes:
            predicted_positions[p4] += w4 * -f4 * delta_lagrange
    
    @staticmethod
    def neo_hookean_constraint(
        predicted_positions, reference_positions, p1, p2, p3, p4,
        youngs_modulus, poisson_ratio, lagrange, dt, masses, fixed_nodes=None, alpha=0.0
    ):
        """
        Enforce Neo-Hookean elasticity constraint for a tetrahedron.

        Parameters:
        - predicted_positions: Numpy array of predicted vertex positions.
        - reference_positions: Numpy array of reference vertex positions.
        - p1, p2, p3, p4: Indices of the vertices forming the tetrahedron.
        - youngs_modulus: Young's modulus of the material.
        - poisson_ratio: Poisson's ratio of the material.
        - lagrange: Lagrange multiplier (scalar).
        - dt: Time step size (scalar).
        - masses: Array of vertex masses.
        - fixed_nodes: Set of indices of fixed nodes (optional).
        - alpha: Regularization parameter.

        Returns:
        - Updated predicted_positions and lagrange multiplier.
        """
        if fixed_nodes is None:
            fixed_nodes = set()

        # Material properties
        mu = youngs_modulus / (2 * (1 + poisson_ratio))
        lambda_ = youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

        # Reference and predicted deformation matrices
        Dm = np.column_stack((
            reference_positions[p1] - reference_positions[p4],
            reference_positions[p2] - reference_positions[p4],
            reference_positions[p3] - reference_positions[p4]
        ))
        Ds = np.column_stack((
            predicted_positions[p1] - predicted_positions[p4],
            predicted_positions[p2] - predicted_positions[p4],
            predicted_positions[p3] - predicted_positions[p4]
        ))

        # Compute the inverse of Dm
        Dm_inv = np.linalg.inv(Dm)
        F = Ds @ Dm_inv  # Deformation gradient

        # Compute strain energy density (psi) and first Piola-Kirchhoff stress (P)
        F2 = F.T @ F
        J = np.linalg.det(F)
        logJ = np.log(J)
        I1 = np.trace(F2)

        psi = 0.5 * mu * (I1 - 3) - mu * logJ + 0.5 * lambda_ * logJ**2
        FinvT = np.linalg.inv(F).T
        P = mu * (F - FinvT) + lambda_ * logJ * FinvT

        # Signed volume of the reference tetrahedron
        V0 = abs(np.linalg.det(Dm)) / 6.0

        # Elastic potential gradient (forces)
        H = -V0 * P @ Dm_inv.T
        f1 = H[:, 0]
        f2 = H[:, 1]
        f3 = H[:, 2]
        f4 = -(f1 + f2 + f3)

        # Inverse masses
        w1 = 1.0 / masses[p1] if p1 not in fixed_nodes else 0.0
        w2 = 1.0 / masses[p2] if p2 not in fixed_nodes else 0.0
        w3 = 1.0 / masses[p3] if p3 not in fixed_nodes else 0.0
        w4 = 1.0 / masses[p4] if p4 not in fixed_nodes else 0.0

        # Weighted sum of gradients
        weighted_sum = (
            w1 * np.dot(f1, f1) +
            w2 * np.dot(f2, f2) +
            w3 * np.dot(f3, f3) +
            w4 * np.dot(f4, f4)
        )

        # Check for small gradients to prevent division by zero
        epsilon = 1e-20
        if weighted_sum < epsilon:
            return predicted_positions, lagrange

        # Update Lagrange multiplier and positions
        C = V0 * psi
        alpha_tilde = alpha / (dt**2)
        delta_lagrange = -(C + alpha_tilde * lagrange) / (weighted_sum + alpha_tilde)
        lagrange += delta_lagrange

        # Update positions
        predicted_positions[p1] += w1 * -f1 * delta_lagrange
        predicted_positions[p2] += w2 * -f2 * delta_lagrange
        predicted_positions[p3] += w3 * -f3 * delta_lagrange
        predicted_positions[p4] += w4 * -f4 * delta_lagrange

        return predicted_positions

class VolumeConstraint:
    @staticmethod
    def volume_constraint(
        predicted_positions, reference_positions, p1, p2, p3, p4,
        volume, lagrange, dt, masses, fixed_nodes=None, bc_nodes=None
    ):
        """
        Enforce a volume conservation constraint for a tetrahedron.

        Parameters:
        - predicted_positions: Numpy array of predicted vertex positions.
        - reference_positions: Numpy array of reference vertex positions.
        - p1, p2, p3, p4: Indices of the vertices forming the tetrahedron.
        - volume: The rest volume of the tetrahedron.
        - lagrange: Lagrange multiplier (scalar, updated in-place).
        - dt: Time step size (scalar).
        - masses: Array of vertex masses.
        - fixed_nodes: Set of indices of fixed nodes (optional).
        """
        if fixed_nodes is None:
            fixed_nodes = set()

        # Extract positions
        x1 = predicted_positions[p1]
        x2 = predicted_positions[p2]
        x3 = predicted_positions[p3]
        x4 = predicted_positions[p4]

        # Calculate current volume
        current_volume = np.abs((1.0 / 6.0) * np.dot(np.cross(x2 - x1, x3 - x1), x4 - x1))
        constraint_value = current_volume - volume

        # Calculate gradients
        grad0 = (1.0 / 6.0) * np.cross(x2 - x3, x4 - x3)
        grad1 = (1.0 / 6.0) * np.cross(x3 - x1, x4 - x1)
        grad2 = (1.0 / 6.0) * np.cross(x1 - x2, x4 - x2)
        grad3 = (1.0 / 6.0) * np.cross(x2 - x1, x3 - x1)

        # Compute inverse masses
        w0 = 0.0 if p1 in (fixed_nodes or bc_nodes) else 1.0 / masses[p1]
        w1 = 0.0 if p2 in (fixed_nodes or bc_nodes) else 1.0 / masses[p2]
        w2 = 0.0 if p3 in (fixed_nodes or bc_nodes) else 1.0 / masses[p3]
        w3 = 0.0 if p4 in (fixed_nodes or bc_nodes) else 1.0 / masses[p4]

        # Weighted sum of gradient magnitudes
        weighted_sum_of_gradients = (
            w0 * np.dot(grad0, grad0) +
            w1 * np.dot(grad1, grad1) +
            w2 * np.dot(grad2, grad2) +
            w3 * np.dot(grad3, grad3)
        )

        # Avoid numerical issues if gradients are too small
        if weighted_sum_of_gradients < 1e-5:
            return

        # Calculate Lagrange multiplier correction
        alpha_tilde = 1.0 / (dt**2)
        delta_lagrange = -(constraint_value + alpha_tilde * lagrange) / (
            weighted_sum_of_gradients + alpha_tilde
        )

        # Update Lagrange multiplier
        lagrange += delta_lagrange

        # Apply position corrections
        predicted_positions[p1] += w0 * grad0 * delta_lagrange
        predicted_positions[p2] += w1 * grad1 * delta_lagrange
        predicted_positions[p3] += w2 * grad2 * delta_lagrange
        predicted_positions[p4] += w3 * grad3 * delta_lagrange
