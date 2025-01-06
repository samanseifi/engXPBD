class EdgeConstraints:

    @staticmethod
    def distance_constraint(predicted_positions, p1, p2, rest_length, fixed_nodes, masses):
        """Enforce distance constraint between two points with mass weighting."""
        delta = predicted_positions[p2] - predicted_positions[p1]
        delta_length = np.linalg.norm(delta)

        if delta_length - rest_length > 1e-3:
            correction = 0.5 * (delta_length - rest_length) * (delta / delta_length) 

            # Mass-weighted correction
            total_mass = masses[p1] + masses[p2]
            weight_p1 = masses[p1] / total_mass
            weight_p2 = masses[p2] / total_mass

            if p1 not in fixed_nodes:
                predicted_positions[p1] += weight_p2 * correction
            if p2 not in fixed_nodes:
                predicted_positions[p2] -= weight_p1 * correction