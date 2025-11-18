# test_terrain_spheres.py
import os
import numpy as np
import warp as wp
import warp.sim
import warp.sim.render

# Import your terrain generator
import sys
sys.path.append(os.path.dirname(__file__))

class TerrainSphereTest:
    def __init__(self, stage_path="terrain_sphere_test.usd"):
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        
        # Generate terrain
        print("Generating terrain...")
        volume, vertices, indices = wp.sim.generate_terrain(
            primitive_count=15,
            primitive_size=1.2,
            primitive_height=0.6,
            softmin_k=10.0
        )
        
        # Store for visualization
        self.terrain_vertices = vertices
        self.terrain_indices = indices
        
        # Create model builder
        builder = wp.sim.ModelBuilder()
        
        # Sphere parameters
        self.sphere_radius = 0.15
        self.sphere_mass = 1.0
        
        # Add 9 spheres in a 3x3 grid above the terrain
        spacing = 0.5
        start_height = 2.0
        
        for i in range(3):
            for j in range(3):
                x = (i - 1) * spacing  # -0.5, 0, 0.5
                z = (j - 1) * spacing
                
                # Add a rigid body for each sphere
                body_idx = builder.add_body(
                    origin=wp.transform(
                        wp.vec3(x, start_height, z),
                        wp.quat_identity()
                    ),
                    m=self.sphere_mass
                )
                
                # Add sphere shape to this body
                builder.add_shape_sphere(
                    body=body_idx,
                    pos=wp.vec3(0.0, 0.0, 0.0),  # Local position
                    radius=self.sphere_radius,
                    density=1000.0,
                    ke=1.0e4,
                    kd=1.0e2,
                    kf=1.0e2,
                    mu=0.5
                )
        
        # Create SDF shape from generated volume
        terrain_sdf = wp.sim.SDF(volume)
        
        builder.add_shape_sdf(
            ke=1.0e4,
            kd=1000.0,
            kf=1000.0,
            mu=0.5,
            sdf=terrain_sdf,
            body=-1,  # Static shape
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat(0.0, 0.0, 0.0, 1.0),
            scale=wp.vec3(1.0, 1.0, 1.0),
        )
        
        # Finalize model
        self.model = builder.finalize()
        self.model.ground = False  # We're using SDF, not ground plane
        self.model.gravity = 10.0

        print(f"\n=== Shape Thickness Values ===")
        thickness_values = self.model.shape_geo.thickness.numpy()
        for i in range(len(thickness_values)):
            shape_type = self.model.shape_geo.type.numpy()[i]
            print(f"Shape {i}: type={shape_type}, thickness={thickness_values[i]:.6f}")
        print("================================\n")

        print(f"\n=== Model Info ===")
        print(f"Bodies: {self.model.body_count}")
        
        print(f"\n=== Model Info ===")
        print(f"Bodies: {self.model.body_count}")
        print(f"Shapes: {self.model.shape_count}")
        print(f"Rigid contact max: {self.model.rigid_contact_max}")
        print("==================\n")
        
        
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        
        # Use FeatherstoneIntegrator (same as your quadruped but without Moreau)
        self.integrator = wp.sim.FeatherstoneIntegrator(self.model)
        
        # Setup renderer
        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=1.0)
        else:
            self.renderer = None
        
        self.use_cuda_graph = False
    
    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            (self.state_0, self.state_1) = (self.state_1, self.state_0)
    
    def step(self):
        with wp.ScopedTimer("step"):
            self.simulate()
        self.sim_time += self.frame_dt
    
    def render(self):
        if self.renderer is None:
            return
        
        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            
            # Render the terrain MESH for visual reference
            self.renderer.render_mesh(
                name="terrain_mesh",
                points=self.terrain_vertices.tolist(),
                indices=self.terrain_indices.flatten().tolist(),
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                scale=(1.0, 1.0, 1.0)
            )
            
            # Render spheres
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="Warp device")
    parser.add_argument("--stage_path", type=str, default="terrain_sphere_test.usd", 
                        help="USD output path")
    parser.add_argument("--num_frames", type=int, default=300, help="Number of frames")
    
    args = parser.parse_args()
    
    with wp.ScopedDevice(args.device):
        example = TerrainSphereTest(stage_path=args.stage_path)
        
        for i in range(args.num_frames):
            example.step()
            example.render()
            
            if i % 30 == 0:
                print(f"Frame {i}/{args.num_frames}")
        
        if example.renderer:
            example.renderer.save()
            print(f"Saved USD to {args.stage_path}")