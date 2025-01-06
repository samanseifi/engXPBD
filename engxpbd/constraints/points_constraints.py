import numpy as np

class PointConstraints:
    
    @staticmethod
    def fix_points(predicted_positions, reference_positions, fixed_nodes):
        """Fix the positions of specified nodes."""
        for node_idx in fixed_nodes:
            predicted_positions[node_idx] = np.copy(reference_positions[node_idx])