import numpy as np
from engxpbd.mesh.mesh_writer import MeshWriter


class PhysicsSimulator:
    def __init__(self, mesh, dt=0.01, gravity=None, density=10, num_constraints_iterations=50):
        """Initialize the physics simulator."""
        self.problem_type = "solid"
        self.problem_dimension = 2
        
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
        for cell in self.mesh.get_connectivity():
            for i in range(3):
                p1, p2 = sorted((cell[i], cell[(i + 1) % 3]))
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
        density_map = np.full(len(self.mesh.get_connectivity()), density)
        self.masses = self._compute_masses_from_density(density_map)

    def _compute_masses_from_density(self, density_map):
        """
        Compute masses for each node based on tetrahedron density, accounting for shared nodes.

        Parameters:
        - density_map (ndarray): Density values for each triangles.

        Returns:
        - ndarray: Node-based mass distribution.
        """
        # Initialize masses and tetrahedron count per node
        masses = np.zeros(len(self.mesh.node_coords))
        triangle_counts = np.zeros(len(self.mesh.node_coords))

        # Get triangle connectivity
        triangles = self.mesh.get_connectivity()

        for tri_idx, tri in enumerate(triangles):
            # Extract node coordinates
            coords = self.mesh.node_coords[tri]

            # Compute triangle area using determinant formula
            area = 0.5 * np.abs(np.linalg.det(np.column_stack((coords[1] - coords[0], coords[2] - coords[0]))))
                        
            # Compute triangle mass
            triangle_mass = density_map[tri_idx] * area

            # Distribute mass equally among the four nodes of the triangle
            for node in tri:
                masses[node] += triangle_mass / 3.0
                triangle_counts[node] += 1

        # Normalize by the number of triangles sharing each node
        for i in range(len(masses)):
            if triangle_counts[i] > 0:  # Avoid division by zero
                masses[i] /= triangle_counts[i]

        return masses

    def apply_external_force(self, node_indices, force_vector):
        for node_idx in node_indices:
            if node_idx not in self.fixed_nodes:
                self.velocities[node_idx] += (force_vector / self.masses[node_idx]) * self.dt
                
    def apply_displacement(self, node_indices, displacement_vector):
        for node_idx in node_indices:
            # self.mesh.node_coords[node_idx] += (displacement_vector)
            self.velocities[node_idx] = (displacement_vector / self.dt)

    def apply_gravity(self):
        for i in range(len(self.mesh.node_coords)):
            if i not in self.fixed_nodes:        
                self.velocities[i] += self.gravity * self.dt
                
    def damp_velocities(self, damping_factor=0.9):
        self.velocities *= damping_factor

    def predict_positions(self):
        for i in range(len(self.mesh.node_coords)):
            if i not in self.fixed_nodes:
                self.predicted_positions[i] = self.mesh.node_coords[i] + self.velocities[i] * self.dt

    def correct_positions(self, constraints):
        """Apply constraints iteratively to correct predicted positions.

        Parameters:
        - constraints (list): List of constraint functions to apply.
        """
        converged = False
        iterations = 0
        while not converged or iterations < self.num_constraints_iterations:
            old_positions = np.copy(self.predicted_positions)
            for constraint in constraints:
                constraint(self.predicted_positions)             
            err = (np.linalg.norm(old_positions - self.predicted_positions))
            if not converged and err < 1e-3:
                converged = True
                print("Converged after", iterations, "iterations.")
            iterations += 1
                   
    def update_positions(self):
        for i in range(len(self.mesh.node_coords)):
            if i not in self.fixed_nodes:
                self.velocities[i] = (self.predicted_positions[i] - self.mesh.node_coords[i]) / self.dt
                self.mesh.node_coords[i] = self.predicted_positions[i]

    def set_fixed_nodes(self, fixed_node_indices):
        self.fixed_nodes.update(fixed_node_indices)
        
    def set_bc_nodes(self, bc_node_indices):
        self.bc_nodes.update(bc_node_indices)

    def simulate(self, steps, constraints, elasticity_instance=None, save_path=None):
        """Run the simulation for a given number of steps and save results for each step.

        Parameters:
        - steps (int): Number of simulation steps.
        - constraints (list): List of constraint functions to apply.
        - save_path (str): Path to save the VTK files for each step.
        """              
        for step in range(steps):
            
            print(f"Step {step + 1}/{steps}")
                        
            if elasticity_instance is not None:
                elasticity_instance.reset_lagrange_multipliers()
            
            # self.apply_displacement([3,29,28,1,56,134,132,52,57,135,133,53,7,51,50,5], np.array([0.0, 0.0, -0.01]))

            self.apply_gravity()
            self.damp_velocities()
            self.predict_positions()
            self.correct_positions(constraints)
            self.update_positions()
            
            # Save the mesh for each step if save_path is provided
            if save_path:
                MeshWriter.save_mesh_every_n_steps(
                    mesh_reader=self.mesh,
                    output_directory="output_directory",
                    step=int(step),  # Ensure step is an integer
                    interval=int(100),  # Ensure interval is an integer
                    file_format="vtk"
                )

                