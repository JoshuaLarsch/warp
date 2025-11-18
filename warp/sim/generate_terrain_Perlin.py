#!/usr/bin/env python3
"""
Terrain Generator for Warp Physics with Configurable Roughness
Creates matching NVDB (collision SDF) and mesh with sine or perlin noise.

Usage:
    # From your simulation:
    from generate_terrain import get_terrain_mesh_data
    vertices, indices = get_terrain_mesh_data(
        noise_height=0.5,   # Amplitude 
        noise_period=2.0,   # Wavelength
        noise_type="perlin" # "flat", "sine", or "perlin"
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
MESH_RESOLUTION = 100

OUTPUT_NVDB = "testTerrain0.nvdb"
OUTPUT_USD = "testTerrain0.usd"


# ============================================================================
# HEIGHTFIELD GENERATORS
# ============================================================================

@wp.func
def sine_noise(x: float, z: float, period: float) -> float:
    """2D sine wave pattern"""
    freq = 2.0 * 3.14159265 / period
    return wp.sin(x * freq) * wp.sin(z * freq)


@wp.func  
def perlin_noise(x: float, z: float, period: float) -> float:
    """Multi-octave sine (perlin-like)"""
    freq = 2.0 * 3.14159265 / period
    noise = wp.sin(x * freq) * wp.cos(z * freq) * 0.5
    noise += wp.sin(x * freq * 2.0 + 1.5) * wp.cos(z * freq * 2.0 + 2.3) * 0.25
    noise += wp.sin(x * freq * 4.0 + 3.7) * wp.cos(z * freq * 4.0 + 5.1) * 0.125
    noise += wp.sin(x * freq * 8.0 + 7.3) * wp.cos(z * freq * 8.0 + 9.7) * 0.0625
    return noise


@wp.func
def generate_heightfield(x: float, z: float, noise_height: float, noise_period: float, noise_type: int) -> float:
    """Warp kernel version - used by SDF"""
    if noise_type == 0:
        return 0.0
    elif noise_type == 1:
        return sine_noise(x, z, noise_period) * noise_height
    elif noise_type == 2:
        return perlin_noise(x, z, noise_period) * noise_height
    else:
        return 0.0


def generate_heightfield_numpy(x, z, noise_height, noise_period, noise_type):
    """NumPy version - must match Warp version EXACTLY!"""
    if noise_type == "flat":
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
    elif noise_type == "sine":
        freq = 2.0 * np.pi / noise_period
        return np.sin(x * freq) * np.sin(z * freq) * noise_height
    elif noise_type == "perlin":
        freq = 2.0 * np.pi / noise_period
        noise = np.sin(x * freq) * np.cos(z * freq) * 0.5
        noise += np.sin(x * freq * 2.0 + 1.5) * np.cos(z * freq * 2.0 + 2.3) * 0.25
        noise += np.sin(x * freq * 4.0 + 3.7) * np.cos(z * freq * 4.0 + 5.1) * 0.125
        noise += np.sin(x * freq * 8.0 + 7.3) * np.cos(z * freq * 8.0 + 9.7) * 0.0625
        return noise * noise_height
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")


# ============================================================================
# SDF GENERATION
# ============================================================================

@wp.kernel
def compute_sdf_kernel(
    volume: wp.uint64,
    voxel_size: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    noise_height: float,
    noise_period: float,
    noise_type: int,
):
    i, j, k = wp.tid()
    x_world = origin_x + float(i) * voxel_size
    y_world = origin_y + float(j) * voxel_size
    z_world = origin_z + float(k) * voxel_size
    
    terrain_height = generate_heightfield(x_world, z_world, noise_height, noise_period, noise_type)
    distance = y_world - terrain_height
    
    wp.volume_store_f(volume, i, j, k, distance)


def create_sdf_volume(noise_height=0.0, noise_period=1.0, noise_type="flat"):
    print("Generating SDF volume...")
    noise_type_map = {"flat": 0, "sine": 1, "perlin": 2}
    noise_type_int = noise_type_map.get(noise_type, 0)
    
    half_width = TERRAIN_WIDTH / 2.0
    half_length = TERRAIN_LENGTH / 2.0
    origin_x = -half_width
    origin_z = -half_length
    origin_y = -noise_height - 1.0
    
    terrain_height_range = 2.0 * noise_height + 2.0
    nx = int(TERRAIN_WIDTH / VOXEL_SIZE) + 4
    ny = int(terrain_height_range / VOXEL_SIZE) + 4
    nz = int(TERRAIN_LENGTH / VOXEL_SIZE) + 4
    
    print(f"  Noise: {noise_type}, height={noise_height}, period={noise_period}")
    
    volume = wp.Volume.allocate(
        min=(0, 0, 0),
        max=(nx, ny, nz),
        voxel_size=VOXEL_SIZE,
        translation=(origin_x, origin_y, origin_z),
        device="cuda"
    )
    
    wp.launch(
        kernel=compute_sdf_kernel,
        dim=(nx, ny, nz),
        inputs=[volume.id, VOXEL_SIZE, origin_x, origin_y, origin_z, noise_height, noise_period, noise_type_int],
        device="cuda"
    )
    
    wp.synchronize()
    print("  SDF complete!")
    return volume


# ============================================================================
# MESH GENERATION
# ============================================================================

# ============================================================================
# MESH GENERATION - Extract from SDF using Warp kernel
# ============================================================================

@wp.kernel
def extract_surface_kernel(
    volume: wp.uint64,
    voxel_size: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    mesh_resolution: int,
    terrain_width: float,
    terrain_length: float,
    surface_heights: wp.array(dtype=float),
):
    """
    Extract surface heights by finding SDF zero-crossing for each XZ position.
    """
    i, j = wp.tid()
    
    half_width = terrain_width / 2.0
    half_length = terrain_length / 2.0
    
    x_step = terrain_width / float(mesh_resolution - 1)
    z_step = terrain_length / float(mesh_resolution - 1)
    
    # World position for this XZ grid point
    x_world = -half_width + float(i) * x_step
    z_world = -half_length + float(j) * z_step
    
    # Convert to index space
    x_idx = (x_world - origin_x) / voxel_size
    z_idx = (z_world - origin_z) / voxel_size
    
    # Scan downward to find zero-crossing
    y_world_start = 3.0  # Start above terrain
    y_world_end = -2.0   # End below terrain
    num_samples = 100
    
    surface_y = float(0.0)
    
    for k in range(num_samples - 1):
        # Linear interpolation through Y range
        t = float(k) / float(num_samples - 1)
        y_world = y_world_start + t * (y_world_end - y_world_start)
        y_next_world = y_world_start + (t + 1.0 / float(num_samples - 1)) * (y_world_end - y_world_start)
        
        # Convert to index space
        y_idx = (y_world - origin_y) / voxel_size
        y_next_idx = (y_next_world - origin_y) / voxel_size
        
        # Sample SDF at both positions
        sdf_curr = wp.volume_sample_f(volume, wp.vec3(x_idx, y_idx, z_idx), wp.Volume.LINEAR)
        sdf_next = wp.volume_sample_f(volume, wp.vec3(x_idx, y_next_idx, z_idx), wp.Volume.LINEAR)
        
        # Check for zero-crossing (positive to negative)
        if sdf_curr >= 0.0 and sdf_next < 0.0:
            # Linear interpolation to find exact crossing
            if wp.abs(sdf_curr - sdf_next) > 1.0e-6:
                alpha = sdf_curr / (sdf_curr - sdf_next)
                surface_y = y_world + alpha * (y_next_world - y_world)
            else:
                surface_y = y_world
            break
    
    # Store the surface height
    idx = i * mesh_resolution + j
    surface_heights[idx] = surface_y


def generate_mesh_from_sdf(volume, noise_height):
    """
    Extract mesh surface from SDF by finding zero-crossings.
    Uses Warp kernel to query the volume directly.
    
    Args:
        volume: Warp Volume containing the SDF
        noise_height: Used to determine scan range
        
    Returns:
        vertices: Nx3 numpy array of vertex positions (float32)
        indices: Mx3 numpy array of triangle indices (int32)
    """
    print("  Extracting mesh from SDF...")
    
    half_width = TERRAIN_WIDTH / 2.0
    half_length = TERRAIN_LENGTH / 2.0
    
    # Get volume info
    grid_info = volume.get_grid_info()
    translation = grid_info.translation
    voxel_size = VOXEL_SIZE
    
    # Allocate array for surface heights
    surface_heights = wp.zeros(MESH_RESOLUTION * MESH_RESOLUTION, dtype=float, device="cuda")
    
    # Launch kernel to extract surface points
    wp.launch(
        kernel=extract_surface_kernel,
        dim=(MESH_RESOLUTION, MESH_RESOLUTION),
        inputs=[
            volume.id,
            voxel_size,
            translation[0],
            translation[1],
            translation[2],
            MESH_RESOLUTION,
            TERRAIN_WIDTH,
            TERRAIN_LENGTH,
            surface_heights,
        ],
        device="cuda"
    )
    
    wp.synchronize()
    
    # Copy surface heights back to CPU
    surface_heights_np = surface_heights.numpy()
    
    # Build vertices
    vertices = []
    x_step = TERRAIN_WIDTH / (MESH_RESOLUTION - 1)
    z_step = TERRAIN_LENGTH / (MESH_RESOLUTION - 1)
    
    for i in range(MESH_RESOLUTION):
        for j in range(MESH_RESOLUTION):
            x = -half_width + i * x_step
            z = -half_length + j * z_step
            y = surface_heights_np[i * MESH_RESOLUTION + j]
            vertices.append([x, y, z])
    
    # Generate triangle indices
    indices = []
    for i in range(MESH_RESOLUTION - 1):
        for j in range(MESH_RESOLUTION - 1):
            v0 = i * MESH_RESOLUTION + j
            v1 = i * MESH_RESOLUTION + (j + 1)
            v2 = (i + 1) * MESH_RESOLUTION + (j + 1)
            v3 = (i + 1) * MESH_RESOLUTION + j
            indices.append([v0, v1, v2])
            indices.append([v0, v2, v3])
    
    print(f"  Extracted {len(vertices)} vertices from SDF zero-crossings")
    
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)


# ============================================================================
# PUBLIC API - Single function to generate everything
# ============================================================================

def generate_terrain(noise_height=0.0, noise_period=1.0, noise_type="flat"):
    """
    Generate terrain SDF first, then extract mesh from it.
    This ensures PERFECT alignment - mesh is derived from the actual SDF.
    
    Args:
        noise_height: Amplitude of terrain roughness (0.0=flat, 0.5=gentle, 2.0=rough)
        noise_period: Spatial wavelength in meters (smaller = more frequent bumps)
        noise_type: "flat", "sine", or "perlin"
    
    Returns:
        volume: Warp Volume containing the SDF for collision
        vertices: Nx3 numpy array of mesh vertices (float32)
        indices: Mx3 numpy array of triangle indices (int32)
    
    Example:
        volume, vertices, indices = generate_terrain(0.5, 2.0, "perlin")
        
        # Use for collision
        sdf = wp.sim.SDF(volume)
        builder.add_shape_sdf(sdf=sdf, ...)
        
        # Use for rendering
        renderer.render_mesh("terrain", vertices, indices)
    """
    print(f"Generating terrain: {noise_type}, height={noise_height}m, period={noise_period}m")
    
    # Step 1: Generate SDF volume
    volume = create_sdf_volume(noise_height, noise_period, noise_type)
    
    # Step 2: Extract mesh FROM the SDF (guaranteed alignment!)
    vertices, indices = generate_mesh_from_sdf(volume, noise_height)
    
    print(f"✓ Terrain ready: SDF + {len(vertices)} vertices, {len(indices)} triangles")
    
    return volume, vertices, indices


# ============================================================================
# USD EXPORT (optional)
# ============================================================================

def create_usd_mesh(noise_height=0.0, noise_period=1.0, noise_type="flat"):
    """Generate USD by first creating SDF, then extracting mesh from it"""
    print("Generating USD mesh from SDF...")
    
    # Generate SDF
    volume = create_sdf_volume(noise_height, noise_period, noise_type)
    
    # Extract mesh FROM the SDF
    vertices_np, indices_np = generate_mesh_from_sdf(volume, noise_height)
    
    stage = Usd.Stage.CreateNew(OUTPUT_USD)
    mesh = UsdGeom.Mesh.Define(stage, "/Terrain")
    
    vertices_usd = [Gf.Vec3f(v[0], v[1], v[2]) for v in vertices_np]
    face_vertex_counts = [3] * len(indices_np)
    face_vertex_indices = indices_np.flatten().tolist()
    
    mesh.GetPointsAttr().Set(vertices_usd)
    mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
    mesh.CreateDoubleSidedAttr().Set(False)
    
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Terrain"))
    return stage


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def main():
    # CONFIGURE HERE
    noise_height = 0.5
    noise_period = 2.0
    noise_type = "perlin"
    
    print("=" * 60)
    print("Terrain Generator")
    print("=" * 60)
    
    # Generate everything at once
    volume, vertices, indices = generate_terrain(noise_height, noise_period, noise_type)
    
    # Save SDF
    volume.save_to_nvdb(OUTPUT_NVDB)
    print(f"✓ Saved {OUTPUT_NVDB}")
    
    # Save USD (optional)
    usd_stage = create_usd_mesh(noise_height, noise_period, noise_type)
    usd_stage.Save()
    print(f"✓ Saved {OUTPUT_USD}")
    
    print()
    print("=" * 60)
    print("Usage in simulation:")
    print(f"  volume, vertices, indices = generate_terrain({noise_height}, {noise_period}, '{noise_type}')")
    print("  sdf = wp.sim.SDF(volume)")
    print("  renderer.render_mesh('terrain', vertices, indices)")


if __name__ == "__main__":
    main()