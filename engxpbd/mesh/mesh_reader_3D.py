import numpy as np
import meshio


class MeshReaderGmsh:
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