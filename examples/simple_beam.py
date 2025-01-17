import numpy as np
from engxpbd.mesh.mesh2D import Mesh2D
from engxpbd.simulator import PhysicsSimulator
from engxpbd.constraints.points.points_constraints import PointConstraints
from engxpbd.constraints.solid.two_dimension.solid_elasticity_constraints import SolidElasticity2D, AreaConstraint

if __name__ == "__main__":
    mesh = Mesh2D(x_min=0, x_max=10, y_min=0, y_max=1, mesh_size=0.5, structured=True)
    mesh.save_to_ls_dyna("mesh.k0")
    dt = 0.008
    
                
    # Print a summary of the mesh
    print("Mesh summary:")
    print("Number of nodes:", len(mesh.node_coords))
    print("Number of triangles:", len(mesh.triangles))
    print("Number of boundary nodes:", len(mesh.boundary_nodes))
    print("Number of boundary lines:", len(mesh.boundary_lines))
    

    # # Initialize the physics simulator
    simulator = PhysicsSimulator(mesh, dt=dt, gravity=np.array([0, -9.81]), num_constraints_iterations=50)
  
    # Get top nodes as fixed nodes
    fixed_nodes = set(mesh.boundary_nodes["left"])
    print("Fixed nodes:", fixed_nodes)
    right_nodes = set(mesh.boundary_nodes["right"])
    left_nodes = set(mesh.boundary_nodes["left"])
    
    bc_nodes = set([])
    
    simulator.set_fixed_nodes(fixed_nodes)
    simulator.set_bc_nodes(bc_nodes)
    
    # # Initialize SolidElasticity
    elasticity = SolidElasticity2D(
        youngs_modulus=1e6,
        poisson_ratio=0.35,
        dt=dt,
        masses=simulator.masses
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
    
    # constraints.append(lambda pos, pos_init=mesh.node_coords_init, fixed_ids=left_nodes, dof=0:        
    #                        PointConstraints.fix_points(pos, pos_init, fixed_ids, dof))
    
    # constraints.append(lambda pos, pos_init=mesh.node_coords_init, fixed_ids=right_nodes, dof=0:
    #                        PointConstraints.fix_points(pos, pos_init, fixed_ids, dof))

    # volume constraint
    areaconstraint = AreaConstraint(                
        dt=dt,
        masses=simulator.masses,
    )
    
    areaconstraint.initialize_lagrange_multipliers(mesh.get_connectivity())
    
    for cell in mesh.get_connectivity():
        constraints.append(lambda pos, c=cell:
                        areaconstraint.incompressibility(pos, mesh.node_coords_init, fixed_nodes, bc_nodes,  *c))
    

    # # Simulate over 100 steps and save each step
    simulator.simulate(240, constraints, elasticity_instance=elasticity, save_path="solid_simultiom")
    print("Simulation complete. Mesh files saved for each step.")