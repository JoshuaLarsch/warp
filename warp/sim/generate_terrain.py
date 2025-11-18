#!/usr/bin/env python3
"""
Box-Primitive Terrain Generator with Proper SDF
Uses analytical box SDFs combined via softmin for correct distance fields.

Usage:
    from generate_terrain import generate_terrain
    
    volume, vertices, indices = generate_terrain(
        primitive_count=10,
        primitive_size=1.0,
        primitive_height=0.5,
        softmin_k=10.0
    )
"""

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom, Gf

wp.init()

# ============================================================================
# PARAMETERS
# ============================================================================

TERRAIN_WIDTH = 3.0
TERRAIN_LENGTH = 3.0
VOXEL_SIZE = 0.02
MESH_RESOLUTION = 500

OUTPUT_NVDB = "testTerrain0.nvdb"
OUTPUT_USD = "testTerrain0.usd"


# ============================================================================
# BOX SDF - Analytical Distance Field
# ============================================================================

@wp.func
def box_sdf(point: wp.vec3, center: wp.vec3, half_extents: wp.vec3) -> float:
    """
    Analytical SDF for axis-aligned box.
    Returns true 3D distance to nearest box surface.
    
    Args:
        point: Query point in world space
        center: Box center
        half_extents: Half-widths in each dimension
    
    Returns:
        Signed distance (negative inside, positive outside)
    """
    # Translate to box-local coordinates
    p = point - center
    
    # Distance from point to box surface
    q = wp.abs(p) - half_extents
    
    # Distance is:
    # - Outside: length of vector from point to nearest box surface
    # - Inside: negative distance to nearest wall
    outside_dist = wp.length(wp.max(q, wp.vec3(0.0, 0.0, 0.0)))
    inside_dist = wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
    
    return outside_dist + inside_dist


@wp.func
def softmin(distances: wp.array(dtype=float), count: int, k: float, idx: int) -> float:
    """
    Smooth minimum using log-sum-exp trick.
    Combines multiple SDFs smoothly.
    
    Args:
        distances: Array of SDF values
        count: Number of distances to consider
        k: Sharpness parameter (higher = sharper, lower = smoother)
        idx: Thread index offset into distances array
    
    Returns:
        Smooth minimum of all distances
    """
    # Compute exp(-k * d_i) for all distances
    sum_exp = float(0.0)
    
    for i in range(count):
        d = distances[idx * count + i]
        sum_exp += wp.exp(-k * d)
    
    # softmin = -log(sum(exp(-k*d_i))) / k
    return -wp.log(sum_exp) / k


# ============================================================================
# SDF GENERATION KERNEL
# ============================================================================

@wp.kernel
def compute_box_terrain_sdf_kernel(
    volume: wp.uint64,
    voxel_size: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    box_centers: wp.array(dtype=wp.vec3),
    box_half_extents: wp.array(dtype=wp.vec3),
    num_boxes: int,
    softmin_k: float,
):
    """
    Compute SDF by evaluating all box primitives and combining with softmin.
    """
    i, j, k = wp.tid()
    
    # Convert voxel indices to world coordinates
    x_world = origin_x + float(i) * voxel_size
    y_world = origin_y + float(j) * voxel_size
    z_world = origin_z + float(k) * voxel_size
    
    point = wp.vec3(x_world, y_world, z_world)
    

    # Regular Min (no smoothing) if softmin_k larger than 100.0
    # Evaluate SDF for all boxes
    if softmin_k > 100.0:
        result_dist = float(1e6)  # Start with large value
        
        for box_idx in range(num_boxes):
            center = box_centers[box_idx]
            half_extents = box_half_extents[box_idx]
            
            dist = box_sdf(point, center, half_extents)
            
            # Regular min (we'll do softmin later if needed, but regular min is simpler)
            if dist < result_dist:
                result_dist = dist
    else:
        # Compute softmin over all boxes if  softmin_k <= 100.0
        sum_exp = float(0.0)

        for box_idx in range(num_boxes):
            center = box_centers[box_idx]
            half_extents = box_half_extents[box_idx]
            dist = box_sdf(point, center, half_extents)
            
            sum_exp += wp.exp(-softmin_k * dist)

        # softmin = -log(sum(exp(-k*d))) / k
        result_dist = -wp.log(sum_exp) / softmin_k

    wp.volume_store_f(volume, i, j, k, result_dist)


# ============================================================================
# PRIMITIVE GENERATION
# ============================================================================

def generate_box_primitives(primitive_count, primitive_size, primitive_height,seed):
    """
    Generate box primitives for terrain.
    
    Args:
        primitive_count: Number of random boxes (excluding base)
        primitive_size: Maximum edge length for random boxes
        primitive_height: Maximum top surface height
        seed: Random seed for reproducibility
        
    Returns:
        centers: Nx3 array of box centers
        half_extents: Nx3 array of box half-extents
    """
    np.random.seed(seed)
    
    centers = []
    half_extents = []
    
    # Base box - covers entire terrain, top at y=0, bottom at y=-0.5
    base_center = np.array([0.0, -0.25, 0.0])  # Center at y=-0.25
    base_half_extents = np.array([TERRAIN_WIDTH / 2.0, 0.25, TERRAIN_LENGTH / 2.0])
    centers.append(base_center)
    half_extents.append(base_half_extents)
    
    # Random boxes
    half_width = TERRAIN_WIDTH / 2.0
    half_length = TERRAIN_LENGTH / 2.0
    
    for _ in range(primitive_count):
        # Random XZ position within terrain bounds
        x = np.random.uniform(-half_width, half_width)
        z = np.random.uniform(-half_length, half_length)
        
        # Random top surface height from 0 to primitive_height
        top_y = np.random.uniform(0.0, primitive_height)
        
        # Box extends from -0.5 to top_y
        bottom_y = -0.5
        box_height = top_y - bottom_y
        center_y = (top_y + bottom_y) / 2.0
        
        # Random box size (width and length)
        half_size_x = np.random.uniform(0.1, primitive_size / 2.0)
        half_size_z = np.random.uniform(0.1, primitive_size / 2.0)
        
        center = np.array([x, center_y, z])
        half_extent = np.array([half_size_x, box_height / 2.0, half_size_z])
        
        centers.append(center)
        half_extents.append(half_extent)
    
    return np.array(centers, dtype=np.float32), np.array(half_extents, dtype=np.float32)


# ============================================================================
# SDF VOLUME CREATION
# ============================================================================

def create_box_terrain_sdf(primitive_count, primitive_size, primitive_height, softmin_k,seed = 42):
    """
    Create SDF volume from box primitives.
    """
    print("Generating box primitives...")
    centers, half_extents = generate_box_primitives(
        primitive_count, primitive_size, primitive_height,seed = seed
    )
    
    print(f"  {len(centers)} boxes total (1 base + {primitive_count} random)")
    
    # Upload to GPU
    centers_wp = wp.array(centers, dtype=wp.vec3, device="cuda")
    half_extents_wp = wp.array(half_extents, dtype=wp.vec3, device="cuda")
    
    # Calculate volume bounds
    max_height = primitive_height + 4.0  # Add padding
    min_height = -1.0
    
    half_width = TERRAIN_WIDTH / 2.0
    half_length = TERRAIN_LENGTH / 2.0
    
    origin_x = -half_width
    origin_y = min_height
    origin_z = -half_length
    
    # Calculate voxel dimensions
    nx = int(TERRAIN_WIDTH / VOXEL_SIZE) + 4
    ny = int((max_height - min_height) / VOXEL_SIZE) + 4
    nz = int(TERRAIN_LENGTH / VOXEL_SIZE) + 4
    
    print(f"  Volume: {nx} x {ny} x {nz} voxels")
    
    # Allocate volume
    volume = wp.Volume.allocate(
        min=(0, 0, 0),
        max=(nx, ny, nz),
        voxel_size=VOXEL_SIZE,
        translation=(origin_x, origin_y, origin_z),
        device="cuda"
    )
    
    # ⭐ CRITICAL FIX: Initialize entire volume to large positive values
    # This ensures edges/uninitialized voxels are "far outside" any geometry
    print("  Initializing volume to background value...")
    BACKGROUND_DISTANCE = 100.0  # Large positive = far outside
    
    @wp.kernel
    def initialize_volume(
        volume: wp.uint64,
        background_value: float,
    ):
        i, j, k = wp.tid()
        wp.volume_store_f(volume, i, j, k, background_value)
    
    wp.launch(
        kernel=initialize_volume,
        dim=(nx, ny, nz),
        inputs=[volume.id, BACKGROUND_DISTANCE],
        device="cuda"
    )
    
    print("  Computing SDF from box primitives...")
    
    # Compute actual SDF (will overwrite initialized values)
    wp.launch(
        kernel=compute_box_terrain_sdf_kernel,
        dim=(nx, ny, nz),
        inputs=[
            volume.id,
            VOXEL_SIZE,
            origin_x,
            origin_y,
            origin_z,
            centers_wp,
            half_extents_wp,
            len(centers),
            softmin_k,
        ],
        device="cuda"
    )
    
    wp.synchronize()
    print("  ✓ SDF complete!")
    
    return volume

# ============================================================================
# MESH EXTRACTION FROM SDF
# ============================================================================

@wp.kernel
def extract_surface_kernel(
    volume: wp.uint64,
    mesh_resolution: int,
    terrain_width: float,
    terrain_length: float,
    surface_heights: wp.array(dtype=float),
):
    i, j = wp.tid()
    
    half_width = terrain_width / 2.0
    half_length = terrain_length / 2.0
    
    x_step = terrain_width / float(mesh_resolution - 1)
    z_step = terrain_length / float(mesh_resolution - 1)
    
    x_world = -half_width + float(i) * x_step
    z_world = -half_length + float(j) * z_step
    
    # Scan downward to find zero-crossing
    y_world_start = 3.0
    y_world_end = -2.0
    num_samples = 100
    
    surface_y = float(0.0)
    
    for k in range(num_samples - 1):
        t = float(k) / float(num_samples - 1)
        y_world = y_world_start + t * (y_world_end - y_world_start)
        y_next_world = y_world_start + (t + 1.0/float(num_samples - 1)) * (y_world_end - y_world_start)
        
        # ⭐ USE PROPER API - NOT MANUAL INDEXING
        world_pos_curr = wp.vec3(x_world, y_world, z_world)
        world_pos_next = wp.vec3(x_world, y_next_world, z_world)
        
        idx_curr = wp.volume_world_to_index(volume, world_pos_curr)
        idx_next = wp.volume_world_to_index(volume, world_pos_next)
        
        sdf_curr = wp.volume_sample_f(volume, idx_curr, wp.Volume.LINEAR)
        sdf_next = wp.volume_sample_f(volume, idx_next, wp.Volume.LINEAR)
        
        # Zero-crossing detection
        if sdf_curr >= 0.0 and sdf_next < 0.0:
            if wp.abs(sdf_curr - sdf_next) > 1.0e-6:
                alpha = sdf_curr / (sdf_curr - sdf_next)
                surface_y = y_world + alpha * (y_next_world - y_world)
            else:
                surface_y = y_world
            break
    
    idx = i * mesh_resolution + j
    surface_heights[idx] = surface_y

def extract_mesh_from_sdf(volume):
    """Extract mesh from SDF by finding zero-crossings."""
    print("  Extracting mesh from SDF...")
    
    # Allocate surface heights
    surface_heights = wp.zeros(MESH_RESOLUTION * MESH_RESOLUTION, dtype=float, device="cuda")
    
    # Extract surface - UPDATED: Only 5 parameters now
    wp.launch(
        kernel=extract_surface_kernel,
        dim=(MESH_RESOLUTION, MESH_RESOLUTION),
        inputs=[
            volume.id,           # 1. volume
            MESH_RESOLUTION,     # 2. mesh_resolution
            TERRAIN_WIDTH,       # 3. terrain_width
            TERRAIN_LENGTH,      # 4. terrain_length
            surface_heights,     # 5. surface_heights (output)
        ],
        device="cuda"
    )
    
    wp.synchronize()
    
    # Build mesh (rest stays the same)
    surface_heights_np = surface_heights.numpy()
    
    vertices = []
    half_width = TERRAIN_WIDTH / 2.0
    half_length = TERRAIN_LENGTH / 2.0
    x_step = TERRAIN_WIDTH / (MESH_RESOLUTION - 1)
    z_step = TERRAIN_LENGTH / (MESH_RESOLUTION - 1)
    
    for i in range(MESH_RESOLUTION):
        for j in range(MESH_RESOLUTION):
            x = -half_width + i * x_step
            z = -half_length + j * z_step
            y = surface_heights_np[i * MESH_RESOLUTION + j]
            vertices.append([x, y, z])
    
    # Generate triangles
    indices = []
    for i in range(MESH_RESOLUTION - 1):
        for j in range(MESH_RESOLUTION - 1):
            v0 = i * MESH_RESOLUTION + j
            v1 = i * MESH_RESOLUTION + (j + 1)
            v2 = (i + 1) * MESH_RESOLUTION + (j + 1)
            v3 = (i + 1) * MESH_RESOLUTION + j
            indices.append([v0, v1, v2])
            indices.append([v0, v2, v3])
    
    print(f"  ✓ Extracted {len(vertices)} vertices")
    
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)

# ============================================================================
# PUBLIC API
# ============================================================================

def generate_terrain(
    primitive_count=10,
    primitive_size=1.0,
    primitive_height=0.5,
    softmin_k=10.0,
    seed = 42,
):
    """
    Generate terrain from box primitives with proper SDF.
    
    Args:
        primitive_count: Number of random boxes (excluding base box)
        primitive_size: Maximum edge length for random boxes (meters)
        primitive_height: Maximum height of box tops above ground (meters)
        softmin_k: Softmin sharpness (higher = sharper blending)
    
    Returns:
        volume: Warp Volume with proper distance field SDF
        vertices: Nx3 numpy array
        indices: Mx3 numpy array
        
    Example:
        volume, vertices, indices = generate_terrain(
            primitive_count=20,
            primitive_size=1.5,
            primitive_height=0.8,
            softmin_k=10.0
        )
    """
    print("=" * 60)
    print(f"Generating terrain from {primitive_count} box primitives")
    print(f"  Size: up to {primitive_size}m, Height: up to {primitive_height}m")
    print("=" * 60)
    
    # Generate SDF
    volume = create_box_terrain_sdf(
        primitive_count,
        primitive_size,
        primitive_height,
        softmin_k,
        seed
    )
    
    # Extract mesh
    vertices, indices = extract_mesh_from_sdf(volume)

    
    
    print("✓ Terrain ready!")
    return volume, vertices, indices


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def main():
    # Generate terrain
    volume, vertices, indices = generate_terrain(
        primitive_count=15,
        primitive_size=1.2,
        primitive_height=0.6,
        softmin_k=10.0
    )
    
    # Save NVDB
    volume.save_to_nvdb(OUTPUT_NVDB)
    print(f"✓ Saved {OUTPUT_NVDB}")
    
    print()
    print("Usage in simulation:")
    print("  from generate_terrain import generate_terrain")
    print("  volume, vertices, indices = generate_terrain(15, 1.2, 0.6, 10.0)")


if __name__ == "__main__":
    main()