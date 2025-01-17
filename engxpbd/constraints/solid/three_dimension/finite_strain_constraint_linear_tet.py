import numpy as np
    
class SolidElasticity3D:    
    
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
        
        self.mu_ = youngs_modulus / (2 * (1 + poisson_ratio))
        self.lambda_ = youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        self.min_singular_value = 0.577  # Compression limit
        self.epsilon = 1e-20  # Small numerical threshold

        self.lagrange_multipliers = {}
        self.alpha = 1.0
        
        self.piola_stress = {}
        
    def reset_lagrange_multipliers(self):
        for key in self.lagrange_multipliers:
            self.lagrange_multipliers[key] = 0.0
            
    def initalize_piola_stress(self, connectivity):
        for cell in connectivity:
            self.piola_stress[tuple(cell)] = np.zeros((3, 3))

    def initialize_lagrange_multipliers(self, connectivity):
        for cell in connectivity:
            self.lagrange_multipliers[tuple(cell)] = 0.0
    
    def compute_Dm(self, reference_positions, p1, p2, p3, p4):
        return np.column_stack((
            reference_positions[p1] - reference_positions[p4],
            reference_positions[p2] - reference_positions[p4],
            reference_positions[p3] - reference_positions[p4]
        ))
        
    def compute_Ds(self, predicted_positions, p1, p2, p3, p4):
        return np.column_stack((
            predicted_positions[p1] - predicted_positions[p4],
            predicted_positions[p2] - predicted_positions[p4],
            predicted_positions[p3] - predicted_positions[p4]
        ))
    
    def compute_deformation_gradient(self, Dm, Ds):        
        return Ds @ np.linalg.inv(Dm)
        
    def saint_venant_constraint(self, predicted_positions, reference_positions, fixed_nodes, bc_nodes, p1, p2, p3, p4):
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

        Dm = self.compute_Dm(reference_positions, p1, p2, p3, p4)
        Ds = self.compute_Ds(predicted_positions, p1, p2, p3, p4)

        F = self.compute_deformation_gradient(Dm, Ds)

        U, singular_values, Vt = np.linalg.svd(F, full_matrices=True)
        Fhat = np.diag(singular_values)

        if np.linalg.det(F) < 0:
            Fhat[2, 2] = -Fhat[2, 2]
            U[:, 2] = -U[:, 2]

        Fhat[0, 0] = max(Fhat[0, 0], self.min_singular_value)
        Fhat[1, 1] = max(Fhat[1, 1], self.min_singular_value)
        Fhat[2, 2] = max(Fhat[2, 2], self.min_singular_value)

        I = np.identity(3)
        Ehat = 0.5 * (Fhat.T @ Fhat - I)
        Ehat_trace = np.trace(Ehat)
        Piolahat = Fhat @ (2 * self.mu_ * Ehat + self.lambda_ * Ehat_trace * I)

        E = U @ Ehat @ Vt
        Etrace = np.trace(E)
        psi = self.mu_ * np.sum(E**2) + 0.5 * self.lambda_ * Etrace**2

        Piola = U @ Piolahat @ Vt

        V0 = abs(np.linalg.det(Dm)) / 6.0
        H = -V0 * Piola @ np.linalg.inv(Dm).T
        f1 = H[:, 0]
        f2 = H[:, 1]
        f3 = H[:, 2]
        f4 = -(f1 + f2 + f3)  
    
        w1 = 1.0 / self.masses[p1] if p1 not in (fixed_nodes or bc_nodes) else 0.0
        w2 = 1.0 / self.masses[p2] if p2 not in (fixed_nodes or bc_nodes) else 0.0
        w3 = 1.0 / self.masses[p3] if p3 not in (fixed_nodes or bc_nodes) else 0.0
        w4 = 1.0 / self.masses[p4] if p4 not in (fixed_nodes or bc_nodes) else 0.0

        weighted_sum_of_gradients = (
            w1 * np.dot(f1, f1) +
            w2 * np.dot(f2, f2) +
            w3 * np.dot(f3, f3) +
            w4 * np.dot(f4, f4)
        )

        if weighted_sum_of_gradients < self.epsilon:
            return predicted_positions, lagrange  # No updates

        C = V0 * psi
        alpha_tilde = 1.0 / (self.dt**2)
        delta_lagrange = -(C + alpha_tilde * lagrange) / (weighted_sum_of_gradients + alpha_tilde)
        lagrange += delta_lagrange

        predicted_positions[p1] += w1 * -f1 * delta_lagrange
        predicted_positions[p2] += w2 * -f2 * delta_lagrange
        predicted_positions[p3] += w3 * -f3 * delta_lagrange
        predicted_positions[p4] += w4 * -f4 * delta_lagrange

        self.lagrange_multipliers[key] = lagrange
        
        return predicted_positions, lagrange, 
    
    
    def neo_hookean_constraint(self, predicted_positions, reference_positions, fixed_nodes, bc_nodes, p1, p2, p3, p4):
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

        if fixed_nodes is None:
            fixed_nodes = set()

        Dm = self.compute_Dm(reference_positions, p1, p2, p3, p4)
        Ds = self.compute_Ds(predicted_positions, p1, p2, p3, p4)

        F = self.compute_deformation_gradient(Dm, Ds)

        # Compute strain energy density (psi) and first Piola-Kirchhoff stress (P)
        F2 = F.T @ F
        J = np.linalg.det(F)
        logJ = np.log(J)
        I1 = np.trace(F2)

        psi = 0.5 * self.mu_ * (I1 - 3) - self.mu_ * logJ + 0.5 * self.lambda_ * logJ**2
        FinvT = np.linalg.inv(F).T
        P = self.mu_ * (F - FinvT) + self.lambda_ * logJ * FinvT
        
        self.piola_stress[key] = P

        V0 = abs(np.linalg.det(Dm)) / 6.0

        H = -V0 * P @ np.linalg.inv(Dm).T
        f1 = H[:, 0]
        f2 = H[:, 1]
        f3 = H[:, 2]
        f4 = -(f1 + f2 + f3)
        
        assert np.allclose(f1 + f2 + f3 + f4, np.zeros(3)) # Conservation of momentum
        
        centroid = (predicted_positions[p1] + predicted_positions[p2] + predicted_positions[p3] + predicted_positions[p4]) / 4
        r1 = predicted_positions[p1] - centroid
        r2 = predicted_positions[p2] - centroid
        r3 = predicted_positions[p3] - centroid
        r4 = predicted_positions[p4] - centroid
        torque = np.cross(r1, f1) + np.cross(r2, f2) + np.cross(r3, f3) + np.cross(r4, f4)
        assert np.allclose(torque, np.zeros(3)) # Conservation of angular momentum
        
        w1 = 1.0 / self.masses[p1] if p1 not in (fixed_nodes or bc_nodes) else 0.0
        w2 = 1.0 / self.masses[p2] if p2 not in (fixed_nodes or bc_nodes) else 0.0
        w3 = 1.0 / self.masses[p3] if p3 not in (fixed_nodes or bc_nodes) else 0.0
        w4 = 1.0 / self.masses[p4] if p4 not in (fixed_nodes or bc_nodes) else 0.0

        weighted_sum_of_gradients = (
            w1 * np.dot(f1, f1) +
            w2 * np.dot(f2, f2) +
            w3 * np.dot(f3, f3) +
            w4 * np.dot(f4, f4)
        )

        epsilon = 1e-20
        if weighted_sum_of_gradients < epsilon:
            return predicted_positions, lagrange

        C = V0 * psi
        alpha_tilde = self.alpha / (self.dt**2)
        delta_lagrange = -(C + alpha_tilde * lagrange) / (weighted_sum_of_gradients + alpha_tilde)
        lagrange += delta_lagrange

        predicted_positions[p1] += w1 * -f1 * delta_lagrange
        predicted_positions[p2] += w2 * -f2 * delta_lagrange
        predicted_positions[p3] += w3 * -f3 * delta_lagrange
        predicted_positions[p4] += w4 * -f4 * delta_lagrange
        
        self.lagrange_multipliers[key] = lagrange

        return predicted_positions, lagrange
