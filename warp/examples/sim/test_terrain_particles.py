# test_terrain_particles.py
import os
import numpy as np
import warp as wp
import warp.sim
import warp.sim.render

# Import your terrain generator
import sys
sys.path.append(os.path.dirname(__file__))

class TerrainParticleTest:
    def __init__(self, stage_path="terrain_particle_test.usd"):
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
        
        # Particle parameters
        self.radius = 0.05
        builder.default_particle_radius = self.radius
        
        # Add particles in a grid above the terrain
        # Spread them across the terrain area
        builder.add_particle_grid(
            dim_x=10,
            dim_y=5,
            dim_z=10,
            cell_x=self.radius * 2.5,
            cell_y=self.radius * 2.5,
            cell_z=self.radius * 2.5,
            pos=wp.vec3(0.0, 2.0, 0.0),  # Start 2m above terrain
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            mass=0.1,
            jitter=self.radius * 0.2,
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
        self.model.ground = False
        self.model.particle_kf = 25.0
        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_kf *= 2.0
        
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        
        self.integrator = wp.sim.SemiImplicitIntegrator()
        
        # Setup renderer
        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=1.0)
        else:
            self.renderer = None
        
        self.use_cuda_graph = False  # Disabled for debugging
    
    def simulate(self):
        for _ in range(self.sim_substeps):
            
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            (self.state_0, self.state_1) = (self.state_1, self.state_0)
    
    def step(self):
        with wp.ScopedTimer("step"):
            self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2.0)
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
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
            
            # Render particles
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="Warp device")
    parser.add_argument("--stage_path", type=str, default="terrain_particle_test.usd", 
                        help="USD output path")
    parser.add_argument("--num_frames", type=int, default=300, help="Number of frames")
    
    args = parser.parse_args()
    
    with wp.ScopedDevice(args.device):
        example = TerrainParticleTest(stage_path=args.stage_path)
        
        for i in range(args.num_frames):
            example.step()
            example.render()
            
            if i % 30 == 0:
                print(f"Frame {i}/{args.num_frames}")
        
        if example.renderer:
            example.renderer.save()
            print(f"Saved USD to {args.stage_path}")