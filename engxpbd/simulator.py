import numpy as np
from engxpbd.mesh_writer import MeshWriter


class PhysicsSimulator:
    def __init__(self, mesh, dt=0.01, gravity=np.array([9.81, 0, 0]), density=1, num_constraints_iterations=50):
        """Initialize the physics simulator."""
        self.mesh = mesh
        self.dt = dt
        self.num_constraints_iterations = num_constraints_iterations
        self.gravity = gravity
        self.velocities = np.zeros_like(self.mesh.node_coords)
        self.predicted_positions = np.copy(self.mesh.node_coords)
        self.fixed_nodes = set()
        self.bc_nodes = set()
    
        self.masses = np.zeros(len(self.mesh.node_coords))
        
        self.rest_lengths = self._initialize_rest_lengths()
        self._initialize_masses(density)
        
    def _initialize_rest_lengths(self):
        """
        Compute the initial rest lengths for all edges in the mesh.

        Returns:
        - dict: A dictionary mapping edge pairs to their initial rest lengths.
        """
        rest_lengths = {}
        for cell in self.mesh.get_connectivity("tetra"):
            for i in range(4):
                p1, p2 = sorted((cell[i], cell[(i + 1) % 4]))
                if (p1, p2) not in rest_lengths:
                    rest_lengths[(p1, p2)] = np.linalg.norm(
                        self.mesh.node_coords_init[p2] - self.mesh.node_coords_init[p1]
                    )
        return rest_lengths
    
    def _initialize_masses(self, density):
        """
        Initialize masses for each node based on tetrahedron density.

        Returns:
        - ndarray: Node-based mass distribution.
        """
        density_map = np.full(len(self.mesh.get_connectivity("tetra")), density)
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
        masses = np.zeros(len(self.mesh.node_coords))
        tetrahedron_counts = np.zeros(len(self.mesh.node_coords))

        # Get tetrahedron connectivity
        tetrahedrons = self.mesh.get_connectivity("tetra")

        for tet_idx, tet in enumerate(tetrahedrons):
            # Extract node coordinates
            coords = self.mesh.node_coords[tet]

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
            self.mesh.node_coords[node_idx] += (displacement_vector)
            # self.velocities[node_idx] = (displacement_vector / self.dt)

    def apply_gravity(self):
        """Apply gravity to all nodes except fixed ones."""
        for i in range(len(self.mesh.node_coords)):
            if i not in self.fixed_nodes:
                
                self.velocities[i] += self.gravity * self.dt
                
    def damp_velocities(self, damping_factor=0.9):
        """Damp the velocities of all nodes."""
        self.velocities *= damping_factor

    def predict_positions(self):
        """Predict new positions based on current velocities."""
        for i in range(len(self.mesh.node_coords)):
            if i not in self.fixed_nodes:
                self.predicted_positions[i] = self.mesh.node_coords[i] + self.velocities[i] * self.dt

    def correct_positions(self, constraints):
        """Apply constraints iteratively to correct predicted positions.

        Parameters:
        - constraints (list): List of constraint functions to apply.
        """
        # Converge constraints added
        for _ in range(self.num_constraints_iterations):
            for constraint in constraints:
                constraint(self.predicted_positions)                                        
                    
    def update_positions(self):
        """Update node positions and velocities based on corrected positions."""
        for i in range(len(self.mesh.node_coords)):
            if i not in self.fixed_nodes:
                self.velocities[i] = (self.predicted_positions[i] - self.mesh.node_coords[i]) / self.dt
                self.mesh.node_coords[i] = self.predicted_positions[i]

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
            
            # self.apply_displacement([3,29,28,1,56,134,132,52,57,135,133,53,7,51,50,5], np.array([0.0, 0.0, -0.001]))

            self.apply_gravity()
            self.damp_velocities()
            self.predict_positions()
            self.correct_positions(constraints)
            self.update_positions()
            
            # Save the mesh for each step if save_path is provided
            if save_path:
                MeshWriter.save_mesh(
                    self.mesh,
                    f"{save_path}_step_{step + 1}.vtk"
                )
                