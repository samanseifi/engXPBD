import numpy as np
from engxpbd.mesh.mesh_reader import MeshReaderGmsh2D
from engxpbd.simulator import PhysicsSimulator
from engxpbd.constraints.cloth.edge_constraints import EdgeConstraints
from engxpbd.constraints.cloth.surface_constraints import SurfaceConstraints

# Example usage
if __name__ == "__main__":
    filepath = "data/rectangle_mesh.msh"  # Replace with your Gmsh file path
    mesh_reader = MeshReaderGmsh2D(filepath)
    # mesh_reader.add_perturbation(magnitude=0.00)
    
    dt = 0.01
    
    # Print a summary of the mesh
    mesh_reader_summary = f"Mesh Summary:\nNumber of nodes: {len(mesh_reader.node_coords)}\n" + \
                          "\n".join([f"{k}: {len(v)} elements" for k, v in mesh_reader.connectivity.items()])
    print(mesh_reader_summary)

    # Initialize the physics simulator
    simulator = PhysicsSimulator(mesh_reader,  dt=dt, gravity=np.array([0,0, -9.81]), num_constraints_iterations=100)

    # Fix the four corners (example indices, adjust as necessary)
    fixed_nodes = [0,1,2]  # Replace with actual corner node indices
    simulator.set_fixed_nodes(fixed_nodes)

     # Define constraints for a cloth model
    rest_angle = 0.0  # Example rest angle for bending constraints

    constraints = []
    for cell in mesh_reader.get_connectivity():
        for i in range(3):
            p1, p2 = sorted((cell[i], cell[(i + 1) % 3]))
            rest_length = simulator.rest_lengths[(p1, p2)]
            constraints.append(lambda pos, a=p1, b=p2, r=rest_length:
                               EdgeConstraints.distance_constraint(pos, a, b, r, simulator.fixed_nodes, simulator.masses))

    adjacent_pairs = mesh_reader.get_adjacent_triangles()

    for p1, p2, p3, p4 in adjacent_pairs:
        constraints.append(lambda pos, a=p1, b=p2, c=p3, d=p4: 
                        SurfaceConstraints.bending_constraint(pos, a, b, c, d, rest_angle, fixed_nodes, simulator.masses))



    # Simulate over 100 steps and save each step
    simulator.simulate(300, constraints, save_path="cloth_simulation")
    print("Simulation complete. Mesh files saved for each step.")