import numpy as np
from engxpbd.mesh.mesh_reader import MeshReaderGmsh
from engxpbd.simulator import PhysicsSimulator
from engxpbd.constraints.points.points_constraints import PointConstraints
from engxpbd.constraints.solid.three_dimension.finite_strain_constraint_linear_tet import SolidElasticity3D
from engxpbd.constraints.solid.three_dimension.volume_constraint_linear_tet import VolumeConstraint

if __name__ == "__main__":
    filepath = "data/cuboid_tetrahedral_mesh_1order.msh"  # Replace with your Gmsh file path
    mesh = MeshReaderGmsh(filepath)
    dt = 0.008
                
    # Print a summary of the mesh
    mesh_summary = f"Mesh Summary:\nNumber of nodes: {len(mesh.node_coords)}\n" + \
                          "\n".join([f"{k}: {len(v)} elements" for k, v in mesh.connectivity.items()])
    print(mesh_summary)

    # Initialize the physics simulator
    simulator = PhysicsSimulator(mesh, dt=dt, gravity=np.array([0, 0,  -9.81]), problem_dimension=3, num_constraints_iterations=50)
  
    # Fix the four corners (example indices, adjust as necessary)
    # fixed_nodes = [0, 12, 2, 29, 49, 31, 4, 22, 6]  # Replace with actual corner node indices
    fixed_nodes = [0,17,18,2,54,136,138,58,55,137,139,59,4,39,40,6]
    bc_nodes = [] # [3,29,28,1,56,134,132,52,57,135,133,53,7,51,50,5]
    
    simulator.set_fixed_nodes(fixed_nodes)
    simulator.set_bc_nodes(bc_nodes)
    
    # Initialize SolidElasticity
    elasticity = SolidElasticity3D(
        youngs_modulus=1e6,
        poisson_ratio=0.49,
        dt=dt,
        masses=simulator.masses,
        fixed_nodes=fixed_nodes,
        bc_nodes=bc_nodes
    )

    # Initialize Lagrange multipliers
    elasticity.initialize_lagrange_multipliers(mesh.get_connectivity())
    
    constraints = []
    for cell in mesh.get_connectivity():
        constraints.append(lambda pos, c=cell:
                        elasticity.neo_hookean_constraint(pos, mesh.node_coords_init, fixed_nodes, bc_nodes, *c))
    # fixed nodes
    constraints.append(lambda pos, pos_init=mesh.node_coords_init, fixed_ids=fixed_nodes, dof=None:
                           PointConstraints.fix_points(pos, pos_init, fixed_ids, dof))
    
    # constraints.append(lambda pos, pos_init=mesh.node_coords_init, fixed_ids=bc_nodes, dof=1:        
    #                        PointConstraints.fix_points(pos, pos_init, fixed_ids, dof))
    
    # constraints.append(lambda pos, pos_init=mesh.node_coords_init, fixed_ids=bc_nodes, dof=2:
    #                        PointConstraints.fix_points(pos, pos_init, fixed_ids, dof))

    # volume constraint
    # volumeconstraint = VolumeConstraint(                
    #     dt=dt,
    #     masses=simulator.masses,
    #     fixed_nodes=fixed_nodes,
    #     bc_nodes=bc_nodes
    # )
    
    # volumeconstraint.initialize_lagrange_multipliers(mesh.get_connectivity())
    
    # for cell in mesh.get_connectivity():
    #     constraints.append(lambda pos, c=cell:
    #                     volumeconstraint.incompressibility(pos, mesh.node_coords_init, fixed_nodes, bc_nodes, *c))
    

    # Simulate over 100 steps and save each step
    simulator.simulate(150, constraints, elasticity_instance=elasticity, save_path="solid_simultiom")
    print("Simulation complete. Mesh files saved for each step.")