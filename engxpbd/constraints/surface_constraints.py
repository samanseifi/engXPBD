import numpy as np

class SurfaceConstraints:
                    
    @staticmethod
    def bending_constraint(predicted_positions, p1, p2, p3, p4, rest_angle, fixed_nodes, masses):
        """
        Enforce bending constraint between two connected triangles.

        Parameters:
        - predicted_positions (ndarray): Predicted positions of nodes (shape: [n, 3]).
        - p1, p2, p3, p4 (int): Indices of the four nodes forming two connected triangles.
        - rest_angle (float): Rest angle (in radians) between the two edges.

        Returns:
        - corrections (dict): Position corrections for each node as {node_index: correction_vector}.
        """
        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm > 1e-6 else np.zeros_like(v)

        def safe_cross(v1, v2):
            return np.cross(v1, v2) if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6 else np.zeros(3)


        # Retrieve positions
        x1 = predicted_positions[p1]
        x2 = predicted_positions[p2]
        x3 = predicted_positions[p3]
        x4 = predicted_positions[p4]

        # Compute normals of the two triangles
        n1 = normalize(safe_cross(x2, x3))
        n2 = normalize(safe_cross(x2, x4))

        # Dot product of normals and dihedral angle
        d = np.dot(n1, n2)
        d = np.clip(d, -1.0, 1.0)  # Clamp to avoid numerical issues
        current_angle = np.arccos(d)

        # Compute the constraint violation
        C = current_angle - rest_angle

        if abs(C) < 1e-6:  # No significant violation, return zero corrections
            return {p1: np.zeros(3), p2: np.zeros(3), p3: np.zeros(3), p4: np.zeros(3)}

        # Gradients of the bending constraint
        q3 = (safe_cross(x2, n2) + safe_cross(n1, x2) * d) / np.linalg.norm(safe_cross(x2, x3))
        q4 = (safe_cross(x2, n1) + safe_cross(n2, x2) * d) / np.linalg.norm(safe_cross(x2, x4))
        q2 = (-safe_cross(x3, n2) + safe_cross(n1, x3) * d) / np.linalg.norm(safe_cross(x2, x3))  - (safe_cross(x4, n1) + safe_cross(n2, x4) * d) / np.linalg.norm(safe_cross(x2, x4))
        q1 = -q2 - q3 - q4

        # Weights (inverse masses) - assume uniform for simplicity
        w1 = 1.0/masses[p1]
        w2 = 1.0/masses[p2]
        w3 = 1.0/masses[p3]
        w4 = 1.0/masses[p4]        
        w_sum = w1 + w2 + w3 + w4

        # Compute the scaling factor
        scale = - (np.sqrt(1.0 - d * d) * C)  / (np.linalg.norm(q1)**2 + np.linalg.norm(q2)**2 + np.linalg.norm(q3)**2 + np.linalg.norm(q4)**2)

        if p1 not in fixed_nodes:
            predicted_positions[p1] += -((4 * w1 ) / w_sum) * scale * q1 
        if p2 not in fixed_nodes:
            predicted_positions[p2] += -((4 * w2 ) / w_sum) * scale * q2
        if p3 not in fixed_nodes:
            predicted_positions[p3] += -((4 * w3 ) / w_sum) * scale * q3
        if p4 not in fixed_nodes:
            predicted_positions[p4] += -((4 * w4 ) / w_sum) * scale * q4