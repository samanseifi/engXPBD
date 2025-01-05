import meshio
import numpy as np
from collections import defaultdict

class Mesh:
    def __init__(self, filepath):
        """Initialize the mesh reader by loading the Gmsh file."""
        self.filepath = filepath
        self.mesh = None
        self.node_coords = None
        self.node_coords_init = None
        self.node_ids = None
        self.connectivity = {}
        self.node_sets = {}
        self._read_mesh()
        self._store_init_node_coords()

    def _read_mesh(self):
        """Reads the mesh file and populates nodes, connectivity, and node sets."""
        self.mesh = meshio.read(self.filepath)
        self.node_coords = self.mesh.points
        self.node_ids = list(range(len(self.node_coords)))
        for cell_block in self.mesh.cells:
            self.connectivity[cell_block.type] = cell_block.data
        if self.mesh.point_data:
            for name, data in self.mesh.point_data.items():
                self.node_sets[name] = data
    
    def _store_init_node_coords(self):
        """Store the initial node coordinates for perturbations."""
        self.node_coords_init = np.copy(self.node_coords)
                
    def add_perturbation(self, magnitude=0.01):
        """Add a random perturbation to node coordinates."""
        self.node_coords += np.random.uniform(-magnitude, magnitude, self.node_coords.shape)
    
    def get_node_coords(self):
        """Return the node coordinates."""
        return self.node_coords

    def get_node_ids(self):
        """Return the node IDs."""
        return self.node_ids
        
    def get_connectivity(self, cell_type=None):
        """
        Return the connectivity matrix for a given cell type.

        Parameters:
        - cell_type (str): The type of cells (e.g., 'triangle', 'tetra').
                          If None, all connectivity matrices are returned.

        Returns:
        - dict or ndarray: Connectivity matrices.
        """
        if cell_type:
            return self.connectivity.get(cell_type, None)
        return self.connectivity
    

    def get_node_sets(self):
        """Return the node sets."""
        return self.node_sets

class PhysicsSimulator:
    def __init__(self, mesh_reader, dt=0.01, gravity=np.array([9.81, 0, 0]), density=0.1):
        """Initialize the physics simulator."""
        self.mesh_reader = mesh_reader
        self.dt = dt
        self.gravity = gravity
        self.velocities = np.zeros_like(self.mesh_reader.node_coords)
        self.predicted_positions = np.copy(self.mesh_reader.node_coords)
        self.fixed_nodes = set()
        self.bc_nodes = set()
        self.masses = np.zeros(len(self.mesh_reader.node_coords))
        self.rest_lengths = self._initialize_rest_lengths()
        self._initialize_masses(density)
        
    def _initialize_rest_lengths(self):
        """
        Compute the initial rest lengths for all edges in the mesh.

        Returns:
        - dict: A dictionary mapping edge pairs to their initial rest lengths.
        """
        rest_lengths = {}
        for cell in self.mesh_reader.get_connectivity("tetra"):
            for i in range(4):
                p1, p2 = sorted((cell[i], cell[(i + 1) % 4]))
                if (p1, p2) not in rest_lengths:
                    rest_lengths[(p1, p2)] = np.linalg.norm(
                        self.mesh_reader.node_coords_init[p2] - self.mesh_reader.node_coords_init[p1]
                    )
        return rest_lengths
    
    
    def _initialize_masses(self, density):
        """
        Initialize masses for each node based on tetrahedron density.

        Returns:
        - ndarray: Node-based mass distribution.
        """
        density_map = np.full(len(self.mesh_reader.get_connectivity("tetra")), density)
        self.masses = self._compute_masses_from_density(density_map)

    def _compute_masses_from_density(self, density_map):
        """
        Compute masses for each node based on tetrahedron density, accounting for shared nodes.

        Parameters:
        - density_map (ndarray): Density values for each tetrahedron.

        Returns:
        - ndarray: Node-based mass distribution.
        """
        # Initialize masses and tetrahedron count per node
        masses = np.zeros(len(self.mesh_reader.node_coords))
        tetrahedron_counts = np.zeros(len(self.mesh_reader.node_coords))

        # Get tetrahedron connectivity
        tetrahedrons = self.mesh_reader.get_connectivity("tetra")

        for tet_idx, tet in enumerate(tetrahedrons):
            # Extract node coordinates
            coords = self.mesh_reader.node_coords[tet]

            # Compute tetrahedron volume using determinant formula
            volume = np.abs(np.linalg.det(np.array([
                coords[1] - coords[0],
                coords[2] - coords[0],
                coords[3] - coords[0]
            ])) / 6.0)

            # Compute tetrahedron mass
            tetrahedron_mass = density_map[tet_idx] * volume

            # Distribute mass equally among the four nodes of the tetrahedron
            for node in tet:
                masses[node] += tetrahedron_mass / 4.0
                tetrahedron_counts[node] += 1

        # Normalize by the number of tetrahedrons sharing each node
        for i in range(len(masses)):
            if tetrahedron_counts[i] > 0:  # Avoid division by zero
                masses[i] /= tetrahedron_counts[i]

        return masses

    def apply_external_force(self, node_indices, force_vector):
        """
        Apply an external force to specified nodes.

        Parameters:
        - node_indices (list): List of node indices where the force will be applied.
        - force_vector (ndarray): A 3D vector representing the force to be applied.
        """
        for node_idx in node_indices:
            if node_idx not in self.fixed_nodes:
                self.velocities[node_idx] += (force_vector / self.masses[node_idx]) * self.dt
                

    def apply_displacement(self, node_indices, displacement_vector):
        """
        Apply a prescribed displacement to specified nodes.

        Parameters:
        - node_indices (list): List of node indices to apply the displacement.
        - displacement_vector (ndarray): A 3D vector representing the displacement to apply.
        """
        for node_idx in node_indices:
            # self.predicted_positions[node_idx] = (self.mesh_reader.node_coords[node_idx] + displacement_vector)
            self.velocities[node_idx] = (displacement_vector / self.dt)

    
    def apply_gravity(self):
        """Apply gravity to all nodes except fixed ones."""
        for i in range(len(self.mesh_reader.node_coords)):
            if i not in self.fixed_nodes:
                
                self.velocities[i] += self.gravity * self.dt
                
    def damp_velocities(self, damping_factor=0.9):
        """Damp the velocities of all nodes."""
        self.velocities *= damping_factor

    def predict_positions(self):
        """Predict new positions based on current velocities."""
        for i in range(len(self.mesh_reader.node_coords)):
            if i not in self.fixed_nodes:
                self.predicted_positions[i] = self.mesh_reader.node_coords[i] + self.velocities[i] * self.dt

    def correct_positions(self, constraints):
        """Apply constraints iteratively to correct predicted positions.

        Parameters:
        - constraints (list): List of constraint functions to apply.
        """
        # Converge constraints added
        for _ in range(50):
            for constraint in constraints:
                constraint(self.predicted_positions)                                        
                    
    def update_positions(self):
        """Update node positions and velocities based on corrected positions."""
        for i in range(len(self.mesh_reader.node_coords)):
            if i not in self.fixed_nodes:
                self.velocities[i] = (self.predicted_positions[i] - self.mesh_reader.node_coords[i]) / self.dt
                self.mesh_reader.node_coords[i] = self.predicted_positions[i]

    def set_fixed_nodes(self, fixed_node_indices):
        """Fix the nodes specified by their indices.

        Parameters:
        - fixed_node_indices (list): List of node indices to fix.
        """
        self.fixed_nodes.update(fixed_node_indices)
        
    def set_bc_nodes(self, bc_node_indices):
        """Fix the nodes specified by their indices.

        Parameters:
        - fixed_node_indices (list): List of node indices to fix.
        """
        self.bc_nodes.update(bc_node_indices)

    def simulate(self, steps, constraints, save_path=None):
        """Run the simulation for a given number of steps and save results for each step.

        Parameters:
        - steps (int): Number of simulation steps.
        - constraints (list): List of constraint functions to apply.
        - save_path (str): Path to save the VTK files for each step.
        """        
        for step in range(steps):
            
            self.apply_displacement([3,29,28,1,56,134,132,52,57,135,133,53,7,51,50,5], np.array([0.0, 0.0, -0.001]))

            # self.apply_gravity()
            self.damp_velocities()
            self.predict_positions()
            self.correct_positions(constraints)
            self.update_positions()
            
            # Save the mesh for each step if save_path is provided
            if save_path:
                MeshWriter.save_mesh(
                    self.mesh_reader,
                    f"{save_path}_step_{step + 1}.vtk"
                )

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
        if p1 not in fixed_nodes or p1 not in bc_nodes:
            predicted_positions[p1] += w1 * -f1 * delta_lagrange
        if p2 not in fixed_nodes or p2 not in bc_nodes:
            predicted_positions[p2] += w2 * -f2 * delta_lagrange
        if p3 not in fixed_nodes or p3 not in bc_nodes:
            predicted_positions[p3] += w3 * -f3 * delta_lagrange
        if p4 not in fixed_nodes or p4 not in bc_nodes:
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

            
class MeshWriter:
    @staticmethod
    def save_mesh(mesh_reader, output_filepath, file_format="vtk"):
        """
        Save the mesh to a file.

        Parameters:
        - output_filepath (str): Path to save the mesh.
        - file_format (str): Format to save the mesh (default is VTK).
        """
        meshio.write_points_cells(
            output_filepath,
            mesh_reader.node_coords,
            mesh_reader.mesh.cells,
            file_format=file_format
        )

# Example usage
if __name__ == "__main__":
    filepath = "cuboid_tetrahedral_mesh.msh"  # Replace with your Gmsh file path
    mesh = Mesh(filepath)
            
    # Print a summary of the mesh
    mesh_summary = f"Mesh Summary:\nNumber of nodes: {len(mesh.node_coords)}\n" + \
                          "\n".join([f"{k}: {len(v)} elements" for k, v in mesh.connectivity.items()])
    print(mesh_summary)

    # Initialize the physics simulator
    simulator = PhysicsSimulator(mesh)
  
    # Fix the four corners (example indices, adjust as necessary)
    # fixed_nodes = [0, 12, 2, 29, 49, 31, 4, 22, 6]  # Replace with actual corner node indices
    fixed_nodes = [0,17,18,2,54,136,138,58,55,137,139,59,4,39,40,6]
    bc_nodes = [3,29,28,1,56,134,132,52,57,135,133,53,7,51,50,5]
    # fixed_nodes = [0,27,28,29,30,2,104,436,440,444,448,112,105,437,441,445,449,113,106,438,442,446,450,114,107,439,443,447,451,115,4,73,74,75,76,6]
    simulator.set_fixed_nodes(fixed_nodes)
    simulator.set_bc_nodes(bc_nodes)
    
    # Define constraint for solid model
    constraints = []
    for cell in mesh.get_connectivity("tetra"):
        constraints.append(lambda pos, c=cell, m=simulator.masses, f=simulator.fixed_nodes, b=bc_nodes:
                           SolidElasticity.saint_venant_constraint(pos, mesh.node_coords_init, *c, 1e6, 0.49, 0.0, simulator.dt, m, f, b))

    # higher-order constraints
    # constraints = []
    
    # shape_gradients = np.array([
    #     [-1, -1, -1],
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]
    # ])
    # quadrature_weights = np.array([1/6, 1/6, 1/6, 1/6])
    
    # for cell in mesh_reader.get_connectivity("tetra"):
    #     constraints.append(lambda pos, c=cell, r=simulator.rest_lengths, m=simulator.masses, f=simulator.fixed_nodes, s=shape_gradients, q=quadrature_weights:
    #                        SolidSaintVenantElasticity.saint_venant_constraint_high_order(pos, mesh_reader.node_coords_init, c, 1e6, 0.3, 0.0, simulator.dt, m, s, q, f))
    

    # Simulate over 100 steps and save each step
    simulator.simulate(30, constraints, save_path="solid_simultiom")
    print("Simulation complete. Mesh files saved for each step.")

