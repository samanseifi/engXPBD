import numpy as np
import meshio
from collections import defaultdict



class MeshReaderGmsh3D:
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
        
    def get_connectivity(self):
        """
        Return the connectivity matrix for a given cell type.

        Parameters:
        - cell_type (str): The type of cells (e.g., 'triangle', 'tetra').
                          If None, all connectivity matrices are returned.

        Returns:
        - dict or ndarray: Connectivity matrices.
        """    
        cell_type='tetra'
        if cell_type:
            return self.connectivity.get(cell_type, None)
        return self.connectivity

    
    def get_node_sets(self):
        """Return the node sets."""
        return self.node_sets
    
    def get_surface_nodes(self):
        """Return the surface nodes."""
        return self.node_sets.get("surface", None)


class MeshReaderGmsh2D:
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
        
    def get_connectivity(self):
        """
        Return the connectivity matrix for a given cell type.

        Parameters:
        - cell_type (str): The type of cells (e.g., 'triangle', 'tetra').
                          If None, all connectivity matrices are returned.

        Returns:
        - dict or ndarray: Connectivity matrices.
        """
        cell_type='triangle'
        if cell_type:
            return self.connectivity.get(cell_type, None)
        return self.connectivity
    
    def get_connectivity_by_type(self, cell_type):
        """Return the connectivity matrix for a given cell type."""
        return self.connectivity.get(cell_type, None)
    
    def get_adjacent_triangles(self):
        """Find adjacent triangles for bending constraints."""
        adjacency = {}
        triangles = self.get_connectivity_by_type("triangle")
        edges = self.get_connectivity_by_type("line")
        
        if edges is None or triangles is None:
            raise ValueError("The mesh does not contain edges or triangles.")
        
        # Map edges to adjacent triangles
        edge_to_triangles = defaultdict(list)

        # Helper function to sort edges
        def sorted_edge(a, b):
            return tuple(sorted([a, b]))

        # Build edge-to-triangle mapping
        for i, tri in enumerate(triangles):
            for edge in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                edge_to_triangles[sorted_edge(*edge)].append(i)

        # Find [p1, p2, p3, p4] for each shared edge
        edge_tri_node_map = []

        for edge, adjacent_triangles in edge_to_triangles.items():
            if len(adjacent_triangles) == 2:  # Shared edge
                tri1, tri2 = adjacent_triangles

                # Get nodes of the triangles
                nodes_tri1 = set(triangles[tri1])
                nodes_tri2 = set(triangles[tri2])

                # Find the shared edge nodes (p1, p2)
                p1, p2 = edge

                # Find the other nodes from each triangle
                p3 = list(nodes_tri1 - {p1, p2})[0]
                p4 = list(nodes_tri2 - {p1, p2})[0]
                
                edge_tri_node_map.append([p1, p2, p3, p4])
            
            return edge_tri_node_map

    def get_node_sets(self):
        """Return the node sets."""
        return self.node_sets
