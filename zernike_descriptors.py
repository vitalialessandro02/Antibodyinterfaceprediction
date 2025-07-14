import numpy as np
from scipy.special import sph_harm, factorial
from scipy.spatial import cKDTree
from math import sqrt, pi, gamma
import numba
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
import freesasa
from config import (ZERNIKE_ORDER, PATCH_RADIUS, VOXEL_RESOLUTION, 
                   SOLVENT_PROBE_RADIUS, PHYSICOCHEMICAL_PROPERTIES)

class ProteinSurface:
    def __init__(self, pdb_file):
        self.pdb_file = pdb_file
        self.structure = None
        self.atoms = []
        self.surface = None
        self.load_structure()
        
    def load_structure(self):
        """Load PDB structure using BioPython"""
        parser = PDBParser()
        self.structure = parser.get_structure('antibody', self.pdb_file)
        self.atoms = [atom for atom in self.structure.get_atoms()]
        
    def compute_ses(self):
        """Compute Solvent Excluded Surface using FreeSASA"""
        structure = freesasa.Structure(self.pdb_file)
        result = freesasa.calc(structure)
        
        # Get atoms with their SASA values
        atom_areas = {}
        for i, atom in enumerate(structure.atoms()):
            atom_areas[atom.serial] = result.atomArea(i)
            
        return atom_areas
    
    def voxelize_surface(self, property_index):
        """
        Voxelize the protein surface with a specific property mapped
        """
        # Get the property values for each atom
        property_values = self.map_property_to_atoms(property_index)
        
        # Create a 3D grid
        atom_coords = np.array([atom.get_coord() for atom in self.atoms])
        min_coords = atom_coords.min(axis=0) - 10  # Add padding
        max_coords = atom_coords.max(axis=0) + 10
        
        grid_shape = ((max_coords - min_coords) * VOXEL_RESOLUTION).astype(int)
        grid = np.zeros(grid_shape)
        
        # Create KDTree for efficient nearest neighbor search
        tree = cKDTree(atom_coords)
        
        # Map property values to grid
        for index in np.ndindex(grid.shape):
            voxel_center = min_coords + index / VOXEL_RESOLUTION
            dist, idx = tree.query(voxel_center, k=1)
            if dist < (self.atoms[idx].radius + SOLVENT_PROBE_RADIUS):
                grid[index] = property_values[idx]
                
        return grid

@numba.jit(nopython=True)
def zernike_radial_poly(n, l, r):
    """Compute Zernike radial polynomial R_n^l(r)"""
    R = 0.0
    for k in range((n - l) // 2 + 1):
        numerator = (-1)**k * factorial(n - k)
        denominator = factorial(k) * factorial((n + l) // 2 - k) * factorial((n - l) // 2 - k)
        R += numerator / denominator * r**(n - 2 * k)
    return R

def compute_3dzd(f, max_order=ZERNIKE_ORDER):
    """
    Compute 3D Zernike descriptors up to specified order
    """
    descriptors = []
    
    # Get shape and create spherical coordinate grid
    nx, ny, nz = f.shape
    x, y, z = np.mgrid[0:nx, 0:ny, 0:nz].astype(float)
    
    # Normalize coordinates to [-1, 1] range
    x = 2 * x / (nx - 1) - 1
    y = 2 * y / (ny - 1) - 1
    z = 2 * z / (nz - 1) - 1
    
    # Convert to spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / (r + 1e-10))
    phi = np.arctan2(y, x)
    
    # Only consider points inside the unit sphere
    mask = r <= 1.0
    r_masked = r[mask]
    theta_masked = theta[mask]
    phi_masked = phi[mask]
    f_masked = f[mask]
    
    for n in range(0, max_order + 1):
        for l in range(0, n + 1):
            if (n - l) % 2 != 0:
                continue
            for m in range(-l, l + 1):
                # Compute spherical harmonic
                Ylm = sph_harm(m, l, phi_masked, theta_masked)
                
                # Compute radial polynomial
                Rnl = np.array([zernike_radial_poly(n, l, ri) for ri in r_masked])
                
                # Compute integrand
                integrand = f_masked * Rnl * Ylm.conjugate()
                
                # Approximate integral
                moment = np.sum(integrand) * (2 / nx)**3  # voxel size adjustment
                
                # Normalization constant
                N = sqrt((2 * n + 3) / (3 * pi))
                moment *= N
                
                descriptors.append(moment)
    
    # Convert to magnitude (rotation invariant)
    descriptors = np.array([abs(d) for d in descriptors])
    
    return descriptors

def map_properties_to_surface(pdb_file, property_id):
    """
    Map a specific physicochemical property to the protein surface
    """
    # This would be implemented using a voxelization approach
    # For simplicity, we'll return a placeholder function
    # In practice, you'd use a library like Biopython to parse PDB
    # and compute the SES with properties mapped
    
    # Placeholder - returns a 3D array with property values
    return np.random.rand(64, 64, 64)
def extract_patch(surface_grid, center, radius=PATCH_RADIUS):
    """
    Extract a spherical patch from the protein surface
    """
    # Convert center to grid coordinates
    center_idx = np.round(center * VOXEL_RESOLUTION).astype(int)
    
    # Convert radius to voxel units
    radius_vox = int(radius * VOXEL_RESOLUTION)
    
    # Get patch bounds
    x, y, z = center_idx
    x_min = max(0, x - radius_vox)
    x_max = min(surface_grid.shape[0], x + radius_vox + 1)
    y_min = max(0, y - radius_vox)
    y_max = min(surface_grid.shape[1], y + radius_vox + 1)
    z_min = max(0, z - radius_vox)
    z_max = min(surface_grid.shape[2], z + radius_vox + 1)
    
    # Extract patch
    patch = surface_grid[x_min:x_max, y_min:y_max, z_min:z_max]
    
    # Pad if necessary to maintain consistent size
    pad_width = [(max(0, radius_vox - (x - x_min)), 
                  max(0, radius_vox - (x_max - x - 1))),
                 (max(0, radius_vox - (y - y_min)), 
                  max(0, radius_vox - (y_max - y - 1))),
                 (max(0, radius_vox - (z - z_min)), 
                  max(0, radius_vox - (z_max - z - 1)))]
    
    if any(any(p) for p in pad_width):
        patch = np.pad(patch, pad_width, mode='constant')
    
    return patch

