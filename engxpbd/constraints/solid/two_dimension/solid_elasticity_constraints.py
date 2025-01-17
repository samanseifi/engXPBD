import numpy as np

class SolidElasticity2D:
    def __init__(self, youngs_modulus, poisson_ratio, dt, masses, fixed_nodes=None, bc_nodes=None):
        """
        Initialize the SolidElasticity2D instance with material properties and simulation parameters.

        Parameters:
        - youngs_modulus: Young's modulus of the material.
        - poisson_ratio: Poisson's ratio of the material.
        - dt: Time step size.
        - masses: Array of node masses.
        - fixed_nodes: Set of indices of fixed nodes (optional).
        - bc_nodes: Set of indices of boundary condition nodes (optional).

        Only supports linear triangular elements.
        """
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.dt = dt
        self.masses = masses

        self.mu_ = youngs_modulus / (2 * (1 + poisson_ratio))
        self.lambda_ = youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        self.epsilon = 1e-20  # Small numerical threshold

        self.lagrange_multipliers = {}
        self.alpha = 1.0

        self.piola_stress = {}

    def reset_lagrange_multipliers(self):
        for key in self.lagrange_multipliers:
            self.lagrange_multipliers[key] = 0.0

    def initialize_piola_stress(self, connectivity):
        for cell in connectivity:
            self.piola_stress[tuple(cell)] = np.zeros((2, 2))

    def initialize_lagrange_multipliers(self, connectivity):
        for cell in connectivity:
            self.lagrange_multipliers[tuple(cell)] = 0.0

    def compute_Dm(self, reference_positions, p1, p2, p3):
        return np.column_stack((
            reference_positions[p1] - reference_positions[p3],
            reference_positions[p2] - reference_positions[p3]
        ))

    def compute_Ds(self, predicted_positions, p1, p2, p3):
        return np.column_stack((
            predicted_positions[p1] - predicted_positions[p3],
            predicted_positions[p2] - predicted_positions[p3]
        ))

    def compute_deformation_gradient(self, Dm, Ds):        
        return Ds @ np.linalg.inv(Dm)

    def neo_hookean_constraint(self, predicted_positions, reference_positions, fixed_nodes, bc_nodes, p1, p2, p3):
        """
        Enforce the Saint-Venant elasticity constraint for a single triangle.

        Parameters:
        - predicted_positions: Numpy array of predicted vertex positions.
        - reference_positions: Numpy array of reference vertex positions.
        - p1, p2, p3: Indices of the vertices forming the triangle.

        Returns:
        - Tuple: (updated_predicted_positions, updated_lagrange_multiplier)
        """        
        key = (p1, p2, p3)
        lagrange = self.lagrange_multipliers[key]

        if fixed_nodes is None:
            fixed_nodes = set()

        Dm = self.compute_Dm(reference_positions, p1, p2, p3)
        Ds = self.compute_Ds(predicted_positions, p1, p2, p3)

        F = self.compute_deformation_gradient(Dm, Ds)

        # Compute strain energy density (psi) and first Piola-Kirchhoff stress (P)
        F2 = F.T @ F
        J = np.linalg.det(F)
        logJ = np.log(J)
        I1 = np.trace(F2)

        psi = 0.5 * self.mu_ * (I1 - 2) - self.mu_ * logJ + 0.5 * self.lambda_ * logJ**2
        FinvT = np.linalg.inv(F).T
        P = self.mu_ * (F - FinvT) + self.lambda_ * logJ * FinvT

        self.piola_stress[key] = P

        A0 = abs(np.linalg.det(Dm)) / 2.0

        H = -A0 * P @ np.linalg.inv(Dm).T
        f1 = H[:, 0]
        f2 = H[:, 1]
        f3 = -(f1 + f2)

        assert np.allclose(f1 + f2 + f3, np.zeros(2))  # Conservation of momentum

        w1 = 1.0 / self.masses[p1] if p1 not in (fixed_nodes or bc_nodes) else 0.0
        w2 = 1.0 / self.masses[p2] if p2 not in (fixed_nodes or bc_nodes) else 0.0
        w3 = 1.0 / self.masses[p3] if p3 not in (fixed_nodes or bc_nodes) else 0.0

        weighted_sum_of_gradients = (
            w1 * np.dot(f1, f1) +
            w2 * np.dot(f2, f2) +
            w3 * np.dot(f3, f3)
        )

        epsilon = 1e-20
        if weighted_sum_of_gradients < epsilon:
            return predicted_positions, lagrange

        C = A0 * psi
        alpha_tilde = self.alpha / (self.dt**2)
        delta_lagrange = -(C + alpha_tilde * lagrange) / (weighted_sum_of_gradients + alpha_tilde)
        lagrange += delta_lagrange

        predicted_positions[p1] += w1 * -f1 * delta_lagrange
        predicted_positions[p2] += w2 * -f2 * delta_lagrange
        predicted_positions[p3] += w3 * -f3 * delta_lagrange

        self.lagrange_multipliers[key] = lagrange

        return predicted_positions, lagrange
    
class AreaConstraint:
    def __init__(self, dt, masses, fixed_nodes=None, bc_nodes=None):
        self.dt = dt
        self.masses = masses
        
        self.lagrange_multipliers = {}
        self.alpha = 1.0
        
    def reset_lagrange_multipliers(self):
        for key in self.lagrange_multipliers:
            self.lagrange_multipliers[key] = 0.0

    def initialize_lagrange_multipliers(self, connectivity):
        for cell in connectivity:
            self.lagrange_multipliers[tuple(cell)] = 0.0

    def compute_area(self, x1, x2, x3):
        return 0.5 * np.linalg.norm(np.cross(x2 - x1, x3 - x1))

    def incompressibility(self, predicted_positions, reference_positions, fixed_nodes, bc_nodes, p1, p2, p3):
        """
        Enforce an area conservation constraint for a triangle.

        Parameters:
        - predicted_positions: Numpy array of predicted vertex positions.
        - reference_positions: Numpy array of reference vertex positions.
        - p1, p2, p3: Indices of the vertices forming the triangle.
        """

        key = (p1, p2, p3)
        lagrange = self.lagrange_multipliers[key]

        if fixed_nodes is None:
            fixed_nodes = set()

        x1 = predicted_positions[p1]
        x2 = predicted_positions[p2]
        x3 = predicted_positions[p3]

        X1 = reference_positions[p1]
        X2 = reference_positions[p2]
        X3 = reference_positions[p3]

        reference_area = self.compute_area(X1, X2, X3)
        current_area = self.compute_area(x1, x2, x3)

        constraint_value = current_area - reference_area

        grad0 = 0.5 * np.cross(x2 - x3, np.array([0, 0, 1]))[:2]
        grad1 = 0.5 * np.cross(x3 - x1, np.array([0, 0, 1]))[:2]
        grad2 = 0.5 * np.cross(x1 - x2, np.array([0, 0, 1]))[:2]

        w0 = 0.0 if p1 in (fixed_nodes or bc_nodes) else 1.0 / self.masses[p1]
        w1 = 0.0 if p2 in (fixed_nodes or bc_nodes) else 1.0 / self.masses[p2]
        w2 = 0.0 if p3 in (fixed_nodes or bc_nodes) else 1.0 / self.masses[p3]

        weighted_sum_of_gradients = (
            w0 * np.dot(grad0, grad0) + w1 * np.dot(grad1, grad1) +
            w2 * np.dot(grad2, grad2)
        )

        # Avoid numerical issues if gradients are too small
        if weighted_sum_of_gradients < 1e-5:
            return

        alpha_tilde = self.alpha / (self.dt**2)
        delta_lagrange = -(constraint_value + alpha_tilde * lagrange) / (weighted_sum_of_gradients + alpha_tilde)
        lagrange += delta_lagrange

        predicted_positions[p1] += w0 * grad0 * delta_lagrange
        predicted_positions[p2] += w1 * grad1 * delta_lagrange
        predicted_positions[p3] += w2 * grad2 * delta_lagrange

        self.lagrange_multipliers[key] = lagrange

        return predicted_positions, lagrange
