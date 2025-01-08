import numpy as np

class PointConstraints:
    
    @staticmethod
    def fix_points(predicted_positions, reference_positions, fixed_nodes, dof=None,):
        """Fix the positions of specified nodes."""
        if dof is None:
            for node_idx in fixed_nodes:
                predicted_positions[node_idx] = np.copy(reference_positions[node_idx])
        else:
            for node_idx in fixed_nodes:
                predicted_positions[node_idx][dof] = reference_positions[node_idx][dof]
            
    @staticmethod
    def apply_displacement(init_positions, node_indices, displacement_vector, num_iterations):
        """Apply a displacement to specified nodes."""
        for node_idx in node_indices:
            init_positions[node_idx] = displacement_vector * num_iterations