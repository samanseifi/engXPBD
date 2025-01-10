import numpy as np

class SolidElasticity:    
    
    def __init__(self, youngs_modulus, poisson_ratio, dt, masses, fixed_nodes=None, bc_nodes=None):
        """
        Initialize the SolidElasticity instance with material properties and simulation parameters.

        Parameters:
        - youngs_modulus: Young's modulus of the material.
        - poisson_ratio: Poisson's ratio of the material.
        - dt: Time step size.
        - masses: Array of node masses.
        - fixed_nodes: Set of indices of fixed nodes (optional).
        - bc_nodes: Set of indices of boundary condition nodes (optional).
        
        Only supports linear tet elements
        """
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.dt = dt
        self.masses = masses
        self.fixed_nodes = fixed_nodes if fixed_nodes is not None else set()
        self.bc_nodes = bc_nodes if bc_nodes is not None else set()
        
        # Material properties
        self.mu = youngs_modulus / (2 * (1 + poisson_ratio))
        self.lambda_ = youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        self.min_singular_value = 0.577  # Compression limit
        self.epsilon = 1e-20  # Small numerical threshold

        # Dictionary to store Lagrange multipliers for all tetrahedra
        self.lagrange_multipliers = {}
        self.alpha = 1.0
        
    def reset_lagrange_multipliers(self):
        """Reset all Lagrange multipliers to zero."""
        for key in self.lagrange_multipliers:
            self.lagrange_multipliers[key] = 0.0

    def initialize_lagrange_multipliers(self, connectivity):
        """
        Initialize Lagrange multipliers for all tetrahedra in the mesh.

        Parameters:
        - connectivity: Connectivity list of tetrahedra, where each entry contains node indices for a tetrahedron.
        """
        for cell in connectivity:
            self.lagrange_multipliers[tuple(cell)] = 0.0

    def saint_venant_constraint(self, predicted_positions, reference_positions, p1, p2, p3, p4):
        """
        Enforce the Saint-Venant elasticity constraint for a single tetrahedron.

        Parameters:
        - predicted_positions: Numpy array of predicted vertex positions.
        - reference_positions: Numpy array of reference vertex positions.
        - p1, p2, p3, p4: Indices of the vertices forming the tetrahedron.

        Returns:
        - Tuple: (updated_predicted_positions, updated_lagrange_multiplier)
        """
        # Retrieve the Lagrange multiplier for this tetrahedron            
        key = (p1, p2, p3, p4)
        lagrange = self.lagrange_multipliers[key]
        
        # if key == (136, 138, 137, 150):
        #     print(key, lagrange)

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

        # Compute deformation gradient
        Dm_inv = np.linalg.inv(Dm)
        F = Ds @ Dm_inv

        # Singular Value Decomposition (SVD)
        U, singular_values, Vt = np.linalg.svd(F, full_matrices=True)
        Fhat = np.diag(singular_values)

        # Adjust for reflection and compression limit
        if np.linalg.det(F) < 0:
            Fhat[2, 2] = -Fhat[2, 2]
            U[:, 2] = -U[:, 2]

        Fhat[0, 0] = max(Fhat[0, 0], self.min_singular_value)
        Fhat[1, 1] = max(Fhat[1, 1], self.min_singular_value)
        Fhat[2, 2] = max(Fhat[2, 2], self.min_singular_value)

        # Compute strain tensor and stress
        I = np.identity(3)
        Ehat = 0.5 * (Fhat.T @ Fhat - I)
        Ehat_trace = np.trace(Ehat)
        Piolahat = Fhat @ (2 * self.mu * Ehat + self.lambda_ * Ehat_trace * I)

        E = U @ Ehat @ Vt
        Etrace = np.trace(E)
        psi = self.mu * np.sum(E**2) + 0.5 * self.lambda_ * Etrace**2

        Piola = U @ Piolahat @ Vt

        # Compute elastic potential gradient
        V0 = abs(np.linalg.det(Dm)) / 6.0
        H = -V0 * Piola @ Dm_inv.T
        f1 = H[:, 0]
        f2 = H[:, 1]
        f3 = H[:, 2]
        f4 = -(f1 + f2 + f3)  # Enforce momentum conservation
    
        # Compute weights (inverse masses)
        w1 = 1.0 / self.masses[p1] if p1 not in (self.fixed_nodes or self.bc_nodes) else 0.0
        w2 = 1.0 / self.masses[p2] if p2 not in (self.fixed_nodes or self.bc_nodes) else 0.0
        w3 = 1.0 / self.masses[p3] if p3 not in (self.fixed_nodes or self.bc_nodes) else 0.0
        w4 = 1.0 / self.masses[p4] if p4 not in (self.fixed_nodes or self.bc_nodes) else 0.0

        # Compute weighted sum of gradients
        weighted_sum_of_gradients = (
            w1 * np.dot(f1, f1) +
            w2 * np.dot(f2, f2) +
            w3 * np.dot(f3, f3) +
            w4 * np.dot(f4, f4)
        )

        if weighted_sum_of_gradients < self.epsilon:
            return predicted_positions, lagrange  # No updates

        # Update Lagrange multiplier
        C = V0 * psi
        alpha_tilde = 1.0 / (self.dt**2)
        delta_lagrange = -(C + alpha_tilde * lagrange) / (weighted_sum_of_gradients + alpha_tilde)
        lagrange += delta_lagrange

        # Update predicted positions unless fixed or boundary condition nodes
        predicted_positions[p1] += w1 * -f1 * delta_lagrange
        predicted_positions[p2] += w2 * -f2 * delta_lagrange
        predicted_positions[p3] += w3 * -f3 * delta_lagrange
        predicted_positions[p4] += w4 * -f4 * delta_lagrange

        # Update the Lagrange multiplier for this tetrahedron
        self.lagrange_multipliers[key] = lagrange
        
        return predicted_positions, lagrange
    
    def neo_hookean_constraint(self, predicted_positions, reference_positions, p1, p2, p3, p4):
        """
        Enforce the Saint-Venant elasticity constraint for a single tetrahedron.

        Parameters:
        - predicted_positions: Numpy array of predicted vertex positions.
        - reference_positions: Numpy array of reference vertex positions.
        - p1, p2, p3, p4: Indices of the vertices forming the tetrahedron.

        Returns:
        - Tuple: (updated_predicted_positions, updated_lagrange_multiplier)
        """

        key = (p1, p2, p3, p4)
        lagrange = self.lagrange_multipliers[key]

        if self.fixed_nodes is None:
            self.fixed_nodes = set()

        # Material properties
        mu = self.youngs_modulus / (2 * (1 + self.poisson_ratio))
        lambda_ = self.youngs_modulus * self.poisson_ratio / ((1 + self.poisson_ratio) * (1 - 2 * self.poisson_ratio))

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
        w1 = 1.0 / self.masses[p1] if p1 not in (self.fixed_nodes or self.bc_nodes) else 0.0
        w2 = 1.0 / self.masses[p2] if p2 not in (self.fixed_nodes or self.bc_nodes) else 0.0
        w3 = 1.0 / self.masses[p3] if p3 not in (self.fixed_nodes or self.bc_nodes) else 0.0
        w4 = 1.0 / self.masses[p4] if p4 not in (self.fixed_nodes or self.bc_nodes) else 0.0

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
        alpha_tilde = self.alpha / (self.dt**2)
        delta_lagrange = -(C + alpha_tilde * lagrange) / (weighted_sum + alpha_tilde)
        lagrange += delta_lagrange

        # Update positions
        predicted_positions[p1] += w1 * -f1 * delta_lagrange
        predicted_positions[p2] += w2 * -f2 * delta_lagrange
        predicted_positions[p3] += w3 * -f3 * delta_lagrange
        predicted_positions[p4] += w4 * -f4 * delta_lagrange
        
        self.lagrange_multipliers[key] = lagrange

        return predicted_positions, lagrange

class VolumeConstraint:
    
    def __init__(self, dt, masses, fixed_nodes=None, bc_nodes=None):
        self.dt = dt
        self.masses = masses
        self.fixed_nodes = fixed_nodes if fixed_nodes is not None else set()
        self.bc_nodes = bc_nodes if bc_nodes is not None else set()
        
        # Dictionary to store Lagrange multipliers for all tetrahedra
        self.lagrange_multipliers = {}
        self.alpha = 1.0
        
    def reset_lagrange_multipliers(self):
        """Reset all Lagrange multipliers to zero."""
        for key in self.lagrange_multipliers:
            self.lagrange_multipliers[key] = 0.0

    def initialize_lagrange_multipliers(self, connectivity):
        """
        Initialize Lagrange multipliers for all tetrahedra in the mesh.

        Parameters:
        - connectivity: Connectivity list of tetrahedra, where each entry contains node indices for a tetrahedron.
        """
        for cell in connectivity:
            self.lagrange_multipliers[tuple(cell)] = 0.0
    
    def incompressibility(self, predicted_positions, reference_positions, p1, p2, p3, p4):
        """
        Enforce a volume conservation constraint for a tetrahedron.

        Parameters:
        - predicted_positions: Numpy array of predicted vertex positions.
        - reference_positions: Numpy array of reference vertex positions.
        - p1, p2, p3, p4: Indices of the vertices forming the tetrahedron.
        """
       
        key = (p1, p2, p3, p4)
        lagrange = self.lagrange_multipliers[key]
        
        if self.fixed_nodes is None:
            self.sfixed_nodes = set()

        # Extract current positions
        x1 = predicted_positions[p1]
        x2 = predicted_positions[p2]
        x3 = predicted_positions[p3]
        x4 = predicted_positions[p4]
        
        # Extract reference positions
        X1 = reference_positions[p1]
        X2 = reference_positions[p2]
        X3 = reference_positions[p3]
        X4 = reference_positions[p4]

        reference_volume = np.abs((1.0 / 6.0) * np.dot(np.cross(X2 - X1, X3 - X1), X4 - X1))        
        current_volume = np.abs((1.0 / 6.0) * np.dot(np.cross(x2 - x1, x3 - x1), x4 - x1))
        constraint_value = current_volume - reference_volume

        # Calculate gradients
        grad0 = (1.0 / 6.0) * np.cross(x2 - x3, x4 - x3)
        grad1 = (1.0 / 6.0) * np.cross(x3 - x1, x4 - x1)
        grad2 = (1.0 / 6.0) * np.cross(x1 - x2, x4 - x2)
        grad3 = (1.0 / 6.0) * np.cross(x2 - x1, x3 - x1)

        # Compute inverse masses
        w0 = 0.0 if p1 in (self.fixed_nodes or self.bc_nodes) else 1.0 / self.masses[p1]
        w1 = 0.0 if p2 in (self.fixed_nodes or self.bc_nodes) else 1.0 / self.masses[p2]
        w2 = 0.0 if p3 in (self.fixed_nodes or self.bc_nodes) else 1.0 / self.masses[p3]
        w3 = 0.0 if p4 in (self.fixed_nodes or self.bc_nodes) else 1.0 / self.masses[p4]

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
        alpha_tilde = self.alpha / (self.dt**2)
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
        
        self.lagrange_multipliers[key] = lagrange

        return predicted_positions, lagrange
