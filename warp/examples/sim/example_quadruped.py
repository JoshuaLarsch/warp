# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Sim Quadruped
#
# Shows how to set up a simulation of a rigid-body quadruped articulation
# from a URDF using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import math
import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render
import time



# Taken from env/environment.py
def compute_env_offsets(num_envs, env_offset=(5.0, 0.0, 5.0), up_axis="Y"):
    # compute positional offsets per environment
    env_offset = np.array(env_offset)
    nonzeros = np.nonzero(env_offset)[0]
    num_dim = nonzeros.shape[0]
    if num_dim > 0:
        side_length = int(np.ceil(num_envs ** (1.0 / num_dim)))
        env_offsets = []
    else:
        env_offsets = np.zeros((num_envs, 3))
    if num_dim == 1:
        for i in range(num_envs):
            env_offsets.append(i * env_offset)
    elif num_dim == 2:
        for i in range(num_envs):
            d0 = i // side_length
            d1 = i % side_length
            offset = np.zeros(3)
            offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
            offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
            env_offsets.append(offset)
    elif num_dim == 3:
        for i in range(num_envs):
            d0 = i // (side_length * side_length)
            d1 = (i // side_length) % side_length
            d2 = i % side_length
            offset = np.zeros(3)
            offset[0] = d0 * env_offset[0]
            offset[1] = d1 * env_offset[1]
            offset[2] = d2 * env_offset[2]
            env_offsets.append(offset)
    env_offsets = np.array(env_offsets)
    min_offsets = np.min(env_offsets, axis=0)
    correction = min_offsets + (np.max(env_offsets, axis=0) - min_offsets) / 2.0
    if isinstance(up_axis, str):
        up_axis = "XYZ".index(up_axis.upper())
    correction[up_axis] = 0.0  # ensure the envs are not shifted below the ground plane
    env_offsets -= correction
    return env_offsets


class Example:
    def __init__(self, stage_path="example_quadruped.usd", num_envs=8):
        articulation_builder = wp.sim.ModelBuilder()
        wp.sim.parse_urdf(
            os.path.join(warp.examples.get_asset_directory(), "quadruped.urdf"),
            # os.path.join(warp.examples.get_asset_directory(), "quadruped_fixed.urdf"),
            # os.path.join(warp.examples.get_asset_directory(), "test_cube.urdf"),
            # os.path.join(warp.examples.get_asset_directory(), "test_dumbbell.urdf"),
            # os.path.join(warp.examples.get_asset_directory(), "test_sphere.urdf"),
            # os.path.join(warp.examples.get_asset_directory(), "test_tetrahedral.urdf"),
            # os.path.join(warp.examples.get_asset_directory(), "test_corner_cube.urdf"),
            # os.path.join(warp.examples.get_asset_directory(), "test_flapper.urdf"),
            # os.path.join(warp.examples.get_asset_directory(), "test_flapper_fixed.urdf"),
            # os.path.join(warp.examples.get_asset_directory(), "test_pendulum.urdf"),
            # os.path.join(warp.examples.get_asset_directory(), "quadruped_damped.urdf"),
            # os.path.join(warp.examples.get_asset_directory(), "test_3leg.urdf"),
            articulation_builder,
            # xform=wp.transform([0.0, 0.7, 0.0], wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)), # CHANGED TO HAVE CONTACT MUCH EARLIER
            # xform=wp.transform([0.0, 0.7, 0.0], wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)),
            xform=wp.transform([0.0, 0.7, 0.0], wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.48)),
            floating=True,
            density=1000,
            armature=0.01,
            stiffness=200,
            damping=1,
            contact_ke=1.0e4,
            contact_kd=1.0e2,
            contact_kf=1.0e2,
            contact_mu=1.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
        )
        # stage_path = "quadruped_fixed.usd"
        # stage_path = "test_cube.usd"
        # stage_path = "test_dumbbell.usd"
        # stage_path = "test_sphere.usd"
        # stage_path = "test_tetrahedral.usd"
        # stage_path = "test_corner_cube.usd"
        # stage_path = "test_flapper.usd"
        # stage_path = "test_flapper_fixed.usd"
        # tage_path = "test_pendulum.usd"
        # stage_path = "quadruped_damped.usd"
        #stage_path = "test_3leg.usd"
        

        

        builder = wp.sim.ModelBuilder()

        ke = 1.0e4
        kf = 0.0
        kd = 1.0e1
        mu = 0.2

        self.sim_time = 0.0
        fps = 100 # lowered to test instability CHANGED from 100
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 50 # CHANGED from 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs

        offsets = compute_env_offsets(self.num_envs)
        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(offsets[i], wp.quat_identity()))

            builder.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]

            builder.joint_axis_mode = [wp.sim.JOINT_MODE_TARGET_POSITION] * len(builder.joint_axis_mode)
            builder.joint_act[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]

        FilterFootContacts = False
        if FilterFootContacts:
            foot_shape_names = ["LF_SHANK", "RF_SHANK", "LH_SHANK", "RH_SHANK"]
            foot_shape_indices = []

            # Find foot shapes by examining body names
            for shape_idx in range(len(builder.shape_body)):
                body_idx = builder.shape_body[shape_idx]
                if body_idx >= 0:  # Not a static shape.
                    body_name = builder.body_name[body_idx]
                    if any(foot_name in body_name for foot_name in foot_shape_names):
                        foot_shape_indices.append(shape_idx)

            # Disable all shape-shape and ground collisions by default
            for i in range(len(builder.shape_ground_collision)):
                builder.shape_ground_collision[i] = False  # No ground collision
                builder.shape_shape_collision[i] = False   # No shape-shape collision

            # Enable ONLY foot-ground collisions
            for foot_idx in foot_shape_indices:
                builder.shape_ground_collision[foot_idx] = True   # Feet can touch ground
                builder.shape_shape_collision[foot_idx] = True   # But not each other

        # --------------------- GROUND DEFINITION BEGIN ---------------------
        # # BoxGround
        # box_body_idx = builder.body_count  # This will be 13 if you have 13 articulation bodies
        # builder.add_body(
        #     origin=wp.transform(wp.vec3(0.0, -0.25, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.05)),
        #     # You can set mass=0 for static, or leave default
        # )

        # # Now attach the box shape to that body
        # ke, kd, kf, mu = 1.0e4, 1.0e2, 1.0e2, 1.0
        # builder.add_shape_box(
        #     body=box_body_idx,  # Use the body we just created
        #     pos=wp.vec3(0.0, 0.0, 0.0),  # Relative to body origin
        #     hx=2.0, hy=0.25, hz=2.0,
        #     ke=ke, kd=kd, kf=kf, mu=mu
        # )

        # # BoxGround # 2
        # box_body_idx = builder.body_count  # This will be 13 if you have 13 articulation bodies
        # builder.add_body(
        #     origin=wp.transform(wp.vec3(0.0, -0.25, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * -0.05)),
        #     # You can set mass=0 for static, or leave default
        # )

        # # Now attach the box shape to that body
        # ke, kd, kf, mu = 1.0e4, 1.0e2, 1.0e2, 1.0
        # builder.add_shape_box(
        #     body=box_body_idx,  # Use the body we just created
        #     pos=wp.vec3(0.0, 0.0, 0.0),  # Relative to body origin
        #     hx=2.0, hy=0.25, hz=2.0,
        #     ke=ke, kd=kd, kf=kf, mu=mu
        # )

        # print(f"Box body index: {box_body_idx}")

        # # Capsule Ground
        # capsule_body_idx = builder.body_count
        # builder.add_body(
        #     origin=wp.transform(wp.vec3(0.0, -2.0, 0.0), wp.quat_identity()),
        # )

        # ke, kd, kf, mu = 1.0e4, 1.0e2, 1.0e2, 1.0
        # builder.add_shape_capsule(
        #     body=capsule_body_idx,
        #     pos=wp.vec3(0.0, 0.0, 0.0),  # Relative to body origin
        #     radius=2.0,
        #     half_height=20.0,  # Length extends along the axis
        #     up_axis=0,  # 0=X-axis (horizontal), 1=Y-axis (vertical), 2=Z-axis
        #     ke=ke, kd=kd, kf=kf, mu=mu
        # )

        # # SphereGround
        # sphere_body_idx = builder.body_count
        # builder.add_body(
        #     origin=wp.transform(wp.vec3(0.0, -3.0, 0.0), wp.quat_identity()),
        #     # You can set mass=0 for static, or leave default
        # )

        # # Now attach the sphere shape to that body
        # ke, kd, kf, mu = 1.0e4, 1.0e2, 1.0e2, 1.0
        # builder.add_shape_sphere(
        #     body=sphere_body_idx,  # Use the body we just created
        #     pos=wp.vec3(0.0, 0.0, 0.0),  # Relative to body origin
        #     radius=3.0,
        #     ke=ke, kd=kd, kf=kf, mu=mu
        # )

        # print(f"Sphere body index: {sphere_body_idx}")

        # SDFGround ROCK

        # self.rock_path_usd = os.path.join(warp.examples.get_asset_directory(), "rocks.usd")
        # self.rock_path_nvdb = os.path.join(warp.examples.get_asset_directory(), "rocks.nvdb")
        # with open(self.rock_path_nvdb, "rb") as rock_file:
        #     rock_vdb = wp.Volume.load_from_nvdb(rock_file.read())

        # rock_sdf = wp.sim.SDF(rock_vdb)

        # builder.add_shape_sdf(
        #     ke=1.0e4,
        #     kd=1000.0,
        #     kf=1000.0,
        #     mu=0.5,
        #     sdf=rock_sdf,
        #     body=-1,
        #     pos=wp.vec3(1.0, -10.5, 0.5),
        #     # pos=wp.vec3(-0.5, -11, -0.5),
        #     rot=wp.quat(0.0, 0.0, 0.0, 1.0),
        #     scale=wp.vec3(1.0, 1.0, 1.0),
        # )

        # SDFGround Terrain w/ usd and ncdb files

        # self.terrain_path_usd = os.path.join(warp.examples.get_asset_directory(), "testTerrain0.usd")
        # terrain_path_nvdb = os.path.join(warp.examples.get_asset_directory(), "testTerrain0.nvdb")
        # with open(terrain_path_nvdb, "rb") as terrain_file:
        #     terrain_vdb = wp.Volume.load_from_nvdb(terrain_file.read())

        # terrain_sdf = wp.sim.SDF(terrain_vdb)

        # SDFGround Terrain w/generate_terrain.py
          
        # Use for collision
        # volume, self.terrain_vertices, self.terrain_indices = wp.sim.generate_terrain(
        #     noise_height=0.1,
        #     noise_period=2.0,
        #     noise_type="perlin"
        # )

        # volume, self.terrain_vertices, self.terrain_indices = wp.sim.generate_terrain(
        #     primitive_count=15,     # Number of random boxes
        #     primitive_size=2.0,     # Max edge length
        #     primitive_height=0.3,   # Max top height
        #     softmin_k=50.0,        # For future smoothing
        #     seed = 44,
        # )
        # terrain_sdf = wp.sim.SDF(volume)

        # builder.add_shape_sdf(
        #     ke=1.0e4,
        #     kd=1000.0,
        #     kf=1000.0,
        #     mu=0.5,
        #     sdf=terrain_sdf,
        #     body=-1,
        #     pos=wp.vec3(0.0, 0.0, 0.0),
        #     rot=wp.quat(0.0, 0.0, 0.0, 1.0),
        #     # rot=wp.quat_identity(),
        #     scale=wp.vec3(1.0, 1.0, 1.0),
        # )

        # #MeshGround
        # vertices = np.array([
        #     # Bottom vertices (y = -1, below ground)
        #     [-2, -1, -2],  # 0: back-left-bottom
        #     [ 2, -1, -2],  # 1: back-right-bottom  
        #     [ 2, -1,  2],  # 2: front-right-bottom
        #     [-2, -1,  2],  # 3: front-left-bottom
            
        #     # # Top vertices CONCAVE CURVE
        #     # [-2, 0.4, -2], # 4: back-left-top
        #     # [ 2, 0.1, -2], # 5: back-right-top (slightly higher)
        #     # [ 2, 0.5,  2], # 6: front-right-top (highest)
        #     # [-2, 0.1,  2], # 7: front-left-top
        #     # Top vertices CONVEX CURVE
        #     [-2, 0.1, -2], # 4: back-left-top
        #     [ 2, 0.4, -2], # 5: back-right-top
        #     [ 2, 0.1,  2], # 6: front-right-top (lowest)
        #     [-2, 0.5,  2], # 7: front-left-top
        # ], dtype=np.float32)

        # meshShiftX = 0
        # meshShiftY = -0.3
        # for vert in vertices:
        #     vert[0] += meshShiftX
        #     vert[1] += meshShiftY

        # faces = np.array([
        #     # Bottom face (2 triangles)
        #     [0, 1, 2], [0, 2, 3],
        
        #     # Top face (2 triangles, bent)
        #     [4, 7, 6], [4, 6, 5],
            
        #     # Side faces (8 triangles)
        #     [3, 7, 6], [3, 6, 2],  # front face
        #     [0, 4, 5], [0, 5, 1],  # back face  
        #     [0, 3, 7], [0, 7, 4],  # left face
        #     [1, 5, 6], [1, 6, 2],  # right face
        # ], dtype=np.int32)
        # mesh = wp.sim.Mesh(vertices, faces)

        # builder.add_shape_mesh(
        #     body=-1,
        #     pos=wp.vec3(0.0, 0.0, 0.0),
        #     rot=wp.quat_identity(),
        #     mesh=mesh,
        #     ke=ke, kf=kf, kd=kd, mu=mu
        # )

        # --------------------- GROUND DEFINITION  END  ---------------------

        np.set_printoptions(suppress=True)
        # finalize model
        self.model = builder.finalize()
        self.model.ground = True 



        # DEBUG: Print all shapes
        print("\n=== SHAPE INDICES ===")
        shape_bodies = self.model.shape_body.numpy()
        for i in range(len(shape_bodies)):
            body = shape_bodies[i]
            print(f"Shape {i}: body={body}")
        print(f"Total shapes: {len(shape_bodies)}")
        print("=====================\n")

        # self.model.gravity = 0.0

        ############### INCREASE OF CONTACT MARGIN:
        #self.model.soft_contact_margin = 0.5
        #self.model.rigid_contact_margin = 0.2

        self.model.joint_attach_ke = 16000.0
        self.model.joint_attach_kd = 200.0
        self.use_tile_gemm = False
        self.fuse_cholesky = self.use_tile_gemm

        # self.integrator = wp.sim.XPBDIntegrator()
        # self.integrator = wp.sim.SemiImplicitIntegrator()
        # self.integrator = wp.sim.FeatherstoneIntegrator(
        #     self.model, use_tile_gemm=self.use_tile_gemm, fuse_cholesky=self.fuse_cholesky
        # )
        self.integrator = wp.sim.MoreauIntegrator(
            self.model, use_tile_gemm=self.use_tile_gemm, fuse_cholesky=self.fuse_cholesky
        )
        # self.integrator = wp.sim.ClemensMoreauIntegrator(
        #     self.model, use_tile_gemm=self.use_tile_gemm, fuse_cholesky=self.fuse_cholesky
        # )

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path)
        else:
            self.renderer = None

        self.state_0 = self.model.state(requires_grad=True)  
        self.state_1 = self.model.state(requires_grad=True)  
        self.state_mid = self.model.state(requires_grad=True)

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        # simulate() allocates memory via a clone, so we can't use graph capture if the device does not support mempools
        # self.use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())    
        self.use_cuda_graph = True
        if not self.use_cuda_graph:
            print("Graph Capture temporarily disabled for debugging")

        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)


            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            # self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt, state_mid = self.state_mid)
            self.state_0, self.state_1 = self.state_1, self.state_0
            stopVar = 15 # 720 # 84 # 72
            # time.sleep(1)
            # if self.integrator._step > stopVar:
            #     time.sleep(0.5)
            # if self.integrator._step > stopVar + 10:
            #     time.sleep(1000)

    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return
        # self.renderer.render_ref(
        #     name="collision",
        #     path=self.rock_path_usd,
        #     pos=wp.vec3(1.0, -10.5, 0.5),
        #     #pos=wp.vec3(-0.5, -11, -0.5),
        #     rot=wp.quat(0.0, 0.0, 0.0, 1.0),
        #     scale=wp.vec3(1.0, 1.0, 1.0),
        #     color=(0.35, 0.55, 0.9),
        # )
        # self.renderer.render_ref(
        #     name="terrain",
        #     path=self.terrain_path_usd,
        #     pos=wp.vec3(0.0, 0.0, 0.0),
        #     rot=wp.quat(0.0, 0.0, 0.0, 1.0),
        #     scale=wp.vec3(1.0, 1.0, 1.0),
        #     # color=(0.5, 0.4, 0.3),  # Brown-ish terrain color
        #     color=(0.35, 0.55, 0.9),  
        # )
        # self.renderer.render_mesh(
        #     name="terrain",
        #     points=self.terrain_vertices,
        #     indices=self.terrain_indices,
        #     pos=wp.vec3(0.0, 0.0, 0.0),
        #     rot=wp.quat(0.0, 0.0, 0.0, 1.0),
        #     colors=(0.5, 0.4, 0.3),
        # )

        # Visualize contact normals BEGIN
        contact_normals = self.state_0.contact_normals.numpy()
        contact_points = self.state_0.point_vec.numpy()
        c_bodies = self.integrator.c_body_vec.numpy()

        arrow_length = 0.2  # Adjust to taste
        arrow_radius = 0.01

        for env in range(self.num_envs):
            for i in range(4):
                idx = env * 4 + i
                if c_bodies[idx] >= 0:  # Active contact only
                    start = contact_points[idx]
                    normal = contact_normals[idx]
                    end = start + normal * arrow_length
                    
                    # Cylinder position (midpoint)
                    mid = (start + end) / 2.0
                    
                    # Calculate rotation to align cylinder with normal
                    # Default up_axis=1 means cylinder points along Y
                    up = np.array([0, 1, 0])
                    axis = np.cross(up, normal)
                    axis_len = np.linalg.norm(axis)
                    
                    if axis_len > 1e-6:
                        axis = axis / axis_len
                        angle = np.arccos(np.clip(np.dot(up, normal), -1.0, 1.0))
                        # Axis-angle to quaternion (x, y, z, w)
                        s = np.sin(angle / 2)
                        rot = (axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2))
                    else:
                        # Normal parallel to Y - identity or 180° flip
                        rot = (0, 0, 0, 1) if np.dot(up, normal) > 0 else (1, 0, 0, 0)
                    
                    self.renderer.render_cylinder(
                        name=f"normal_{env}_{i}",
                        pos=tuple(mid),
                        rot=rot,
                        radius=arrow_radius,
                        half_height=arrow_length / 2.0,
                        color=(1.0, 0.0, 0.0)  # Red
                    )
        # Visualize contact normals END
        

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_quadruped.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=2, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
