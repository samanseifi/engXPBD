import numpy as np

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