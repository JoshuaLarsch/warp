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
# Example Sim Grad Bounce
#
# Shows how to use Warp to optimize the initial velocity of a particle
# such that it bounces off the wall and floor in order to hit a target.
#
# This example uses the built-in wp.Tape() object to compute gradients of
# the distance to target (loss) w.r.t the initial velocity, followed by
# a simple gradient-descent optimization step.
#
###########################################################################

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render
import os
import warp.examples


@wp.kernel
def loss_kernel(pos: wp.array(dtype=wp.vec3), target: wp.vec3, loss: wp.array(dtype=float)):
    # distance to target
    delta = pos[0] - target
    loss[0] = wp.dot(delta, delta)


@wp.kernel
def step_kernel(x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), alpha: float):
    tid = wp.tid()

    # gradient descent step
    x[tid] = x[tid] - grad[tid] * alpha


class Example:
    def __init__(self, stage_path="example_bounce_various_terains.usd", verbose=False):
        self.verbose = verbose

        # seconds
        sim_duration = 0.6

        # control frequency
        fps = 60
        self.frame_dt = 1.0 / fps
        frame_steps = int(sim_duration / self.frame_dt)

        # sim frequency
        self.sim_substeps = 8
        self.sim_steps = frame_steps * self.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iter = 0
        self.render_time = 0.0

        self.train_rate = 0.02

        ke = 1.0e4
        kf = 0.0
        kd = 1.0e1
        mu = 0.2
        
        self.radius = 0.1

        builder = wp.sim.ModelBuilder()
        builder.add_particle(pos=wp.vec3(-0.5, 1.0, 0.0), vel=wp.vec3(5.0, -5.0, 0.0), mass=1.0,radius=self.radius)
        builder.add_shape_box(body=-1, pos=wp.vec3(2.0, 1.0, 0.0), hx=0.25, hy=1.0, hz=1.0, ke=ke, kf=kf, kd=kd, mu=mu)

        ####################### DIFFERENT COLLISION SURFACE TESTS BEGIN #######################
        # ADD MESH FLOOR
        # mesh_points, mesh_indices = warp.sim.utils.load_mesh("C:/Users/Joshua/Desktop/DiffSimWarpFork/warp/examples/optim/testTerrain01.STL")
        # mesh_points_wp = wp.array(mesh_points, dtype=wp.vec3)
        # mesh_indices_wp = wp.array(mesh_indices, dtype=int)
        # mesh = wp.Mesh(points=mesh_points_wp, indices=mesh_indices_wp)

        # vertices = np.array([[-2,0,-2], [2,0,-2], [2,0.5,2], [-2,0.5,2]], dtype=np.float32)
        # faces = np.array([[0,1,2], [0,2,3]], dtype=np.int32)

        vertices = np.array([
            # Bottom vertices (y = -1, below ground)
            [-2, -1, -2],  # 0: back-left-bottom
            [ 2, -1, -2],  # 1: back-right-bottom  
            [ 2, -1,  2],  # 2: front-right-bottom
            [-2, -1,  2],  # 3: front-left-bottom
            
            # # Top vertices CONCAVE CURVE
            # [-2, 0.4, -2], # 4: back-left-top
            # [ 2, 0.1, -2], # 5: back-right-top (slightly higher)
            # [ 2, 0.5,  2], # 6: front-right-top (highest)
            # [-2, 0.1,  2], # 7: front-left-top
            # Top vertices CONVEX CURVE
            [-2, 0.1, -2], # 4: back-left-top
            [ 2, 0.4, -2], # 5: back-right-top
            [ 2, 0.1,  2], # 6: front-right-top (lowest)
            [-2, 0.5,  2], # 7: front-left-top
        ], dtype=np.float32)

        meshShift = 0.3
        for vert in vertices:
            vert[0] += meshShift

        faces = np.array([
            # Bottom face (2 triangles)
            [0, 1, 2], [0, 2, 3],
            
            # Top face (2 triangles, bent)
            [4, 7, 6], [4, 6, 5],
            
            # Side faces (8 triangles)
            [3, 7, 6], [3, 6, 2],  # front face
            [0, 4, 5], [0, 5, 1],  # back face  
            [0, 3, 7], [0, 7, 4],  # left face
            [1, 5, 6], [1, 6, 2],  # right face
        ], dtype=np.int32)
        mesh = wp.sim.Mesh(vertices, faces)

        builder.add_shape_mesh(
            body=-1,
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            mesh=mesh,
            ke=ke, kf=kf, kd=kd, mu=mu
        )
        # BoxGround
        # builder.add_shape_box(body=-1, pos=wp.vec3(2.0, 0.0, 0.0), hx=1.0, hy=0.25, hz=1.0, ke=ke, kf=kf, kd=kd, mu=mu)

        # SDFGround
        # with open(os.path.join(warp.examples.get_asset_directory(), "rocks.nvdb"), "rb") as rock_file:
        #     rock_vdb = wp.Volume.load_from_nvdb(rock_file.read())

        # rock_sdf = wp.sim.SDF(rock_vdb)

        # builder.add_shape_sdf(
        #     ke=1.0e4,
        #     kd=1000.0,
        #     kf=1000.0,
        #     mu=0.5,
        #     sdf=rock_sdf,
        #     body=-1,
        #     pos=wp.vec3(1.0, -10, 0.5),
        #     rot=wp.quat(0.0, 0.0, 0.0, 1.0),
        #     scale=wp.vec3(1.0, 1.0, 1.0),
        # )

        ####################### DIFFERENT COLLISION SURFACE TESTS  END  #######################

        # use `requires_grad=True` to create a model for differentiable simulation
        self.model = builder.finalize(requires_grad=True)
        self.model.ground = False

        self.model.soft_contact_ke = ke
        self.model.soft_contact_kf = kf
        self.model.soft_contact_kd = kd
        self.model.soft_contact_mu = mu
        self.model.soft_contact_margin = 10.0
        self.model.soft_contact_restitution = 1.0

        # self.integrator = wp.sim.SemiImplicitIntegrator()
        # self.integrator = wp.sim.FeatherstoneIntegrator(self.model)
        self.integrator = wp.sim.MoreauIntegrator(self.model)

        self.target = (-2.0, 1.5, 0.0)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        # allocate sim states for trajectory
        self.states = []
        for _i in range(self.sim_steps + 1):
            self.states.append(self.model.state())

        # one-shot contact creation (valid if we're doing simple collision against a constant normal plane)
        wp.sim.collide(self.model, self.states[0])

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=1.0)
        else:
            self.renderer = None

        # capture forward/backward passes
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(self.loss)
            self.graph = capture.graph

    def forward(self):
        # run control loop
        for i in range(self.sim_steps):
            self.states[i].clear_forces()
            wp.sim.collide(self.model, self.states[i])
            self.integrator.simulate(self.model, self.states[i], self.states[i + 1], self.sim_dt)

        # compute loss on final state
        wp.launch(loss_kernel, dim=1, inputs=[self.states[-1].particle_q, self.target, self.loss])

        return self.loss

    def step(self):
        with wp.ScopedTimer("step"):
            self.model.particle_grid.build(self.states[0].particle_q, self.radius * 2.0) # ADDITION TO FIX COLLISION ISSUES
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.tape = wp.Tape()
                with self.tape:
                    self.forward()
                self.tape.backward(self.loss)

            # gradient descent step
            x = self.states[0].particle_qd
            wp.launch(step_kernel, dim=len(x), inputs=[x, x.grad, self.train_rate])

            x_grad = self.tape.gradients[self.states[0].particle_qd]

            if self.verbose:
                print(f"Iter: {self.iter} Loss: {self.loss}")
                print(f"    x: {x} g: {x_grad}")

            # clear grads for next iteration
            self.tape.zero()

            self.iter = self.iter + 1

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            # draw trajectory
            traj_verts = [self.states[0].particle_q.numpy()[0].tolist()]

            # self.renderer.render_ref(
            #     name="collision",
            #     path=os.path.join(warp.examples.get_asset_directory(), "rocks.usd"),
            #     pos=wp.vec3(1.0, -10, 0.5),
            #     rot=wp.quat(0.0, 0.0, 0.0, 1.0),
            #     scale=wp.vec3(1.0, 1.0, 1.0),
            #     color=(0.35, 0.55, 0.9),
            # )
            for i in range(0, self.sim_steps, self.sim_substeps):
                traj_verts.append(self.states[i].particle_q.numpy()[0].tolist())

                self.renderer.begin_frame(self.render_time)
                self.renderer.render(self.states[i])
                self.renderer.render_box(
                    pos=self.target,
                    rot=wp.quat_identity(),
                    extents=(0.1, 0.1, 0.1),
                    name="target",
                    color=(0.0, 0.0, 0.0),
                )
                self.renderer.render_line_strip(
                    vertices=traj_verts,
                    color=wp.render.bourke_color_map(0.0, 7.0, self.loss.numpy()[0]),
                    radius=0.02,
                    name=f"traj_{self.iter - 1}",
                )
                self.renderer.end_frame()

                from pxr import Gf, UsdGeom

                particles_prim = self.renderer.stage.GetPrimAtPath("/root/particles")
                particles = UsdGeom.Points.Get(self.renderer.stage, particles_prim.GetPath())
                particles.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 1.0, 1.0)], time=self.renderer.time)

                self.render_time += self.frame_dt

    def check_grad(self):
        param = self.states[0].particle_qd

        # initial value
        x_c = param.numpy().flatten()

        # compute numeric gradient
        x_grad_numeric = np.zeros_like(x_c)

        for i in range(len(x_c)):
            eps = 1.0e-3

            step = np.zeros_like(x_c)
            step[i] = eps

            x_1 = x_c + step
            x_0 = x_c - step

            param.assign(x_1)
            l_1 = self.forward().numpy()[0]

            param.assign(x_0)
            l_0 = self.forward().numpy()[0]

            dldx = (l_1 - l_0) / (eps * 2.0)

            x_grad_numeric[i] = dldx

        # reset initial state
        param.assign(x_c)

        # compute analytic gradient
        tape = wp.Tape()
        with tape:
            l = self.forward()

        tape.backward(l)

        x_grad_analytic = tape.gradients[param]

        print(f"numeric grad: {x_grad_numeric}")
        print(f"analytic grad: {x_grad_analytic}")

        tape.zero()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_bounce_various_terains.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--train_iters", type=int, default=250, help="Total number of training iterations.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, verbose=args.verbose)

        example.check_grad()

        # replay and optimize
        for i in range(args.train_iters):
            example.step()
            if i % 16 == 0:
                example.render()

        if example.renderer:
            example.renderer.save()
