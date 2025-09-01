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

import warp as wp
from .model import ModelShapeGeometry

from .articulation import (
    compute_2d_rotational_dofs,
    compute_3d_rotational_dofs,
    eval_fk,
)
from .integrator import Integrator
from .integrator_euler import (
    eval_bending_forces,
    eval_joint_force,
    eval_muscle_forces,
    eval_particle_body_contact_forces,
    eval_particle_forces,
    eval_particle_ground_contact_forces,
    eval_rigid_contacts,
    eval_spring_forces,
    eval_tetrahedral_forces,
    eval_triangle_contact_forces,
    eval_triangle_forces,
)
from .model import Control, Model, State


# Frank & Park definition 3.20, pg 100
@wp.func
def transform_twist(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    w = wp.quat_rotate(q, w)
    v = wp.quat_rotate(q, v) + wp.cross(p, w)

    return wp.spatial_vector(w, v)


@wp.func
def transform_wrench(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    v = wp.quat_rotate(q, v)
    w = wp.quat_rotate(q, w) + wp.cross(p, v)

    return wp.spatial_vector(w, v)


@wp.func
def spatial_adjoint(R: wp.mat33, S: wp.mat33):
    # T = [R  0]
    #     [S  R]

    # fmt: off
    return wp.spatial_matrix(
        R[0, 0], R[0, 1], R[0, 2],     0.0,     0.0,     0.0,
        R[1, 0], R[1, 1], R[1, 2],     0.0,     0.0,     0.0,
        R[2, 0], R[2, 1], R[2, 2],     0.0,     0.0,     0.0,
        S[0, 0], S[0, 1], S[0, 2], R[0, 0], R[0, 1], R[0, 2],
        S[1, 0], S[1, 1], S[1, 2], R[1, 0], R[1, 1], R[1, 2],
        S[2, 0], S[2, 1], S[2, 2], R[2, 0], R[2, 1], R[2, 2],
    )
    # fmt: on


@wp.kernel
def compute_spatial_inertia(
    body_inertia: wp.array(dtype=wp.mat33),
    body_mass: wp.array(dtype=float),
    # outputs
    body_I_m: wp.array(dtype=wp.spatial_matrix),
):
    tid = wp.tid()
    I = body_inertia[tid]
    m = body_mass[tid]
    # fmt: off
    body_I_m[tid] = wp.spatial_matrix(
        I[0, 0], I[0, 1], I[0, 2], 0.0, 0.0, 0.0,
        I[1, 0], I[1, 1], I[1, 2], 0.0, 0.0, 0.0,
        I[2, 0], I[2, 1], I[2, 2], 0.0, 0.0, 0.0,
        0.0,     0.0,     0.0,     m,   0.0, 0.0,
        0.0,     0.0,     0.0,     0.0, m,   0.0,
        0.0,     0.0,     0.0,     0.0, 0.0, m,
    )
    # fmt: on


@wp.kernel
def compute_com_transforms(
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_X_com: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    com = body_com[tid]
    body_X_com[tid] = wp.transform(com, wp.quat_identity())


# computes adj_t^-T*I*adj_t^-1 (tensor change of coordinates), Frank & Park, section 8.2.3, pg 290
@wp.func
def spatial_transform_inertia(t: wp.transform, I: wp.spatial_matrix):
    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.matrix_from_cols(r1, r2, r3)
    S = wp.skew(p) @ R

    T = spatial_adjoint(R, S)

    return wp.mul(wp.mul(wp.transpose(T), I), T)


# compute transform across a joint
@wp.func
def jcalc_transform(
    type: int,
    joint_axis: wp.array(dtype=wp.vec3),
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    joint_q: wp.array(dtype=float),
    start: int,
):
    if type == wp.sim.JOINT_PRISMATIC:
        q = joint_q[start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(axis * q, wp.quat_identity())
        return X_jc

    if type == wp.sim.JOINT_REVOLUTE:
        q = joint_q[start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
        return X_jc

    if type == wp.sim.JOINT_BALL:
        qx = joint_q[start + 0]
        qy = joint_q[start + 1]
        qz = joint_q[start + 2]
        qw = joint_q[start + 3]

        X_jc = wp.transform(wp.vec3(), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == wp.sim.JOINT_FIXED:
        X_jc = wp.transform_identity()
        return X_jc

    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        px = joint_q[start + 0]
        py = joint_q[start + 1]
        pz = joint_q[start + 2]

        qx = joint_q[start + 3]
        qy = joint_q[start + 4]
        qz = joint_q[start + 5]
        qw = joint_q[start + 6]

        X_jc = wp.transform(wp.vec3(px, py, pz), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == wp.sim.JOINT_COMPOUND:
        rot, _ = compute_3d_rotational_dofs(
            joint_axis[axis_start],
            joint_axis[axis_start + 1],
            joint_axis[axis_start + 2],
            joint_q[start + 0],
            joint_q[start + 1],
            joint_q[start + 2],
            0.0,
            0.0,
            0.0,
        )

        X_jc = wp.transform(wp.vec3(), rot)
        return X_jc

    if type == wp.sim.JOINT_UNIVERSAL:
        rot, _ = compute_2d_rotational_dofs(
            joint_axis[axis_start],
            joint_axis[axis_start + 1],
            joint_q[start + 0],
            joint_q[start + 1],
            0.0,
            0.0,
        )

        X_jc = wp.transform(wp.vec3(), rot)
        return X_jc

    if type == wp.sim.JOINT_D6:
        pos = wp.vec3(0.0)
        rot = wp.quat_identity()

        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            pos += axis * joint_q[start + 0]
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            pos += axis * joint_q[start + 1]
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            pos += axis * joint_q[start + 2]

        ia = axis_start + lin_axis_count
        iq = start + lin_axis_count
        if ang_axis_count == 1:
            axis = joint_axis[ia]
            rot = wp.quat_from_axis_angle(axis, joint_q[iq])
        if ang_axis_count == 2:
            rot, _ = compute_2d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_q[iq + 0],
                joint_q[iq + 1],
                0.0,
                0.0,
            )
        if ang_axis_count == 3:
            rot, _ = compute_3d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_axis[ia + 2],
                joint_q[iq + 0],
                joint_q[iq + 1],
                joint_q[iq + 2],
                0.0,
                0.0,
                0.0,
            )

        X_jc = wp.transform(pos, rot)
        return X_jc

    # default case
    return wp.transform_identity()


# compute motion subspace and velocity for a joint
@wp.func
def jcalc_motion(
    type: int,
    joint_axis: wp.array(dtype=wp.vec3),
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    X_sc: wp.transform,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    q_start: int,
    qd_start: int,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
):
    if type == wp.sim.JOINT_PRISMATIC:
        axis = joint_axis[axis_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == wp.sim.JOINT_REVOLUTE:
        axis = joint_axis[axis_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == wp.sim.JOINT_UNIVERSAL:
        axis_0 = joint_axis[axis_start + 0]
        axis_1 = joint_axis[axis_start + 1]
        q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, wp.cross(axis_0, axis_1)))
        local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
        local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))

        axis_0 = local_0
        q_0 = wp.quat_from_axis_angle(axis_0, joint_q[q_start + 0])

        axis_1 = wp.quat_rotate(q_0, local_1)

        S_0 = transform_twist(X_sc, wp.spatial_vector(axis_0, wp.vec3()))
        S_1 = transform_twist(X_sc, wp.spatial_vector(axis_1, wp.vec3()))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1]

    if type == wp.sim.JOINT_COMPOUND:
        axis_0 = joint_axis[axis_start + 0]
        axis_1 = joint_axis[axis_start + 1]
        axis_2 = joint_axis[axis_start + 2]
        q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, axis_2))
        local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
        local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))
        local_2 = wp.quat_rotate(q_off, wp.vec3(0.0, 0.0, 1.0))

        axis_0 = local_0
        q_0 = wp.quat_from_axis_angle(axis_0, joint_q[q_start + 0])

        axis_1 = wp.quat_rotate(q_0, local_1)
        q_1 = wp.quat_from_axis_angle(axis_1, joint_q[q_start + 1])

        axis_2 = wp.quat_rotate(q_1 * q_0, local_2)

        S_0 = transform_twist(X_sc, wp.spatial_vector(axis_0, wp.vec3()))
        S_1 = transform_twist(X_sc, wp.spatial_vector(axis_1, wp.vec3()))
        S_2 = transform_twist(X_sc, wp.spatial_vector(axis_2, wp.vec3()))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1] + S_2 * joint_qd[qd_start + 2]

    if type == wp.sim.JOINT_D6:
        v_j_s = wp.spatial_vector()
        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + 0]
            joint_S_s[qd_start + 0] = S_s
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + 1]
            joint_S_s[qd_start + 1] = S_s
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + 2]
            joint_S_s[qd_start + 2] = S_s
        if ang_axis_count > 0:
            axis = joint_axis[axis_start + lin_axis_count + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 0]
            joint_S_s[qd_start + lin_axis_count + 0] = S_s
        if ang_axis_count > 1:
            axis = joint_axis[axis_start + lin_axis_count + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 1]
            joint_S_s[qd_start + lin_axis_count + 1] = S_s
        if ang_axis_count > 2:
            axis = joint_axis[axis_start + lin_axis_count + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 2]
            joint_S_s[qd_start + lin_axis_count + 2] = S_s

        return v_j_s

    if type == wp.sim.JOINT_BALL:
        S_0 = transform_twist(X_sc, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        S_1 = transform_twist(X_sc, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        S_2 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1] + S_2 * joint_qd[qd_start + 2]

    if type == wp.sim.JOINT_FIXED:
        return wp.spatial_vector()

    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        v_j_s = transform_twist(
            X_sc,
            wp.spatial_vector(
                joint_qd[qd_start + 0],
                joint_qd[qd_start + 1],
                joint_qd[qd_start + 2],
                joint_qd[qd_start + 3],
                joint_qd[qd_start + 4],
                joint_qd[qd_start + 5],
            ),
        )

        joint_S_s[qd_start + 0] = transform_twist(X_sc, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 1] = transform_twist(X_sc, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 2] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 3] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        joint_S_s[qd_start + 4] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        joint_S_s[qd_start + 5] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        return v_j_s

    wp.printf("jcalc_motion not implemented for joint type %d\n", type)

    # default case
    return wp.spatial_vector()


# computes joint space forces/torques in tau
@wp.func
def jcalc_tau(
    type: int,
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_axis_mode: wp.array(dtype=int),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    body_f_s: wp.spatial_vector,
    # outputs
    tau: wp.array(dtype=float),
):
    if type == wp.sim.JOINT_PRISMATIC or type == wp.sim.JOINT_REVOLUTE:
        S_s = joint_S_s[dof_start]

        q = joint_q[coord_start]
        qd = joint_qd[dof_start]
        act = joint_act[axis_start]

        lower = joint_limit_lower[axis_start]
        upper = joint_limit_upper[axis_start]

        limit_ke = joint_limit_ke[axis_start]
        limit_kd = joint_limit_kd[axis_start]
        target_ke = joint_target_ke[axis_start]
        target_kd = joint_target_kd[axis_start]
        mode = joint_axis_mode[axis_start]

        # total torque / force on the joint
        t = -wp.dot(S_s, body_f_s) + eval_joint_force(
            q, qd, act, target_ke, target_kd, lower, upper, limit_ke, limit_kd, mode
        )

        tau[dof_start] = t

        return

    if type == wp.sim.JOINT_BALL:
        # target_ke = joint_target_ke[axis_start]
        # target_kd = joint_target_kd[axis_start]

        for i in range(3):
            S_s = joint_S_s[dof_start + i]

            # w = joint_qd[dof_start + i]
            # r = joint_q[coord_start + i]

            tau[dof_start + i] = -wp.dot(S_s, body_f_s)  # - w * target_kd - r * target_ke

        return

    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        for i in range(6):
            S_s = joint_S_s[dof_start + i]
            tau[dof_start + i] = -wp.dot(S_s, body_f_s)

        return

    if type == wp.sim.JOINT_COMPOUND or type == wp.sim.JOINT_UNIVERSAL or type == wp.sim.JOINT_D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            S_s = joint_S_s[dof_start + i]

            q = joint_q[coord_start + i]
            qd = joint_qd[dof_start + i]
            act = joint_act[axis_start + i]

            lower = joint_limit_lower[axis_start + i]
            upper = joint_limit_upper[axis_start + i]
            limit_ke = joint_limit_ke[axis_start + i]
            limit_kd = joint_limit_kd[axis_start + i]
            target_ke = joint_target_ke[axis_start + i]
            target_kd = joint_target_kd[axis_start + i]
            mode = joint_axis_mode[axis_start + i]

            f = eval_joint_force(q, qd, act, target_ke, target_kd, lower, upper, limit_ke, limit_kd, mode)

            # total torque / force on the joint
            t = -wp.dot(S_s, body_f_s) + f

            tau[dof_start + i] = t

        return


@wp.func
def jcalc_integrate(
    type: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float),
):
    if type == wp.sim.JOINT_FIXED:
        return

    # prismatic / revolute
    if type == wp.sim.JOINT_PRISMATIC or type == wp.sim.JOINT_REVOLUTE:
        qdd = joint_qdd[dof_start]
        qd = joint_qd[dof_start]
        q = joint_q[coord_start]

        qd_new = qd + qdd * dt
        q_new = q + qd_new * dt

        joint_qd_new[dof_start] = qd_new
        joint_q_new[coord_start] = q_new

        return

    # ball
    if type == wp.sim.JOINT_BALL:
        m_j = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        w_j = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        r_j = wp.quat(
            joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2], joint_q[coord_start + 3]
        )

        # symplectic Euler
        w_j_new = w_j + m_j * dt

        drdt_j = wp.quat(w_j_new, 0.0) * r_j * 0.5

        # new orientation (normalized)
        r_j_new = wp.normalize(r_j + drdt_j * dt)

        # update joint coords
        joint_q_new[coord_start + 0] = r_j_new[0]
        joint_q_new[coord_start + 1] = r_j_new[1]
        joint_q_new[coord_start + 2] = r_j_new[2]
        joint_q_new[coord_start + 3] = r_j_new[3]

        # update joint vel
        joint_qd_new[dof_start + 0] = w_j_new[0]
        joint_qd_new[dof_start + 1] = w_j_new[1]
        joint_qd_new[dof_start + 2] = w_j_new[2]

        return

    # free joint
    if type == wp.sim.JOINT_FREE or type == wp.sim.JOINT_DISTANCE:
        # dofs: qd = (omega_x, omega_y, omega_z, vel_x, vel_y, vel_z)
        # coords: q = (trans_x, trans_y, trans_z, quat_x, quat_y, quat_z, quat_w)

        # angular and linear acceleration
        m_s = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        a_s = wp.vec3(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5])

        # angular and linear velocity
        w_s = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])
        v_s = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        # symplectic Euler
        w_s = w_s + m_s * dt
        v_s = v_s + a_s * dt

        # translation of origin
        p_s = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        # linear vel of origin (note q/qd switch order of linear angular elements)
        # note we are converting the body twist in the space frame (w_s, v_s) to compute center of mass velocity
        dpdt_s = v_s + wp.cross(w_s, p_s)

        # quat and quat derivative
        r_s = wp.quat(
            joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
        )

        drdt_s = wp.quat(w_s, 0.0) * r_s * 0.5

        # new orientation (normalized)
        p_s_new = p_s + dpdt_s * dt
        r_s_new = wp.normalize(r_s + drdt_s * dt)

        # update transform
        joint_q_new[coord_start + 0] = p_s_new[0]
        joint_q_new[coord_start + 1] = p_s_new[1]
        joint_q_new[coord_start + 2] = p_s_new[2]

        joint_q_new[coord_start + 3] = r_s_new[0]
        joint_q_new[coord_start + 4] = r_s_new[1]
        joint_q_new[coord_start + 5] = r_s_new[2]
        joint_q_new[coord_start + 6] = r_s_new[3]

        # update joint_twist
        joint_qd_new[dof_start + 0] = w_s[0]
        joint_qd_new[dof_start + 1] = w_s[1]
        joint_qd_new[dof_start + 2] = w_s[2]
        joint_qd_new[dof_start + 3] = v_s[0]
        joint_qd_new[dof_start + 4] = v_s[1]
        joint_qd_new[dof_start + 5] = v_s[2]

        return

    # other joint types (compound, universal, D6)
    if type == wp.sim.JOINT_COMPOUND or type == wp.sim.JOINT_UNIVERSAL or type == wp.sim.JOINT_D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            qdd = joint_qdd[dof_start + i]
            qd = joint_qd[dof_start + i]
            q = joint_q[coord_start + i]

            qd_new = qd + qdd * dt
            q_new = q + qd_new * dt

            joint_qd_new[dof_start + i] = qd_new
            joint_q_new[coord_start + i] = q_new

        return


@wp.func
def compute_link_transform(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    body_X_com: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
):
    # parent transform
    parent = joint_parent[i]
    child = joint_child[i]

    # parent transform in spatial coordinates
    X_pj = joint_X_p[i]
    X_cj = joint_X_c[i]
    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj

    type = joint_type[i]
    axis_start = joint_axis_start[i]
    lin_axis_count = joint_axis_dim[i, 0]
    ang_axis_count = joint_axis_dim[i, 1]
    coord_start = joint_q_start[i]

    # compute transform across joint
    X_j = jcalc_transform(type, joint_axis, axis_start, lin_axis_count, ang_axis_count, joint_q, coord_start)

    # transform from world to joint anchor frame at child body
    X_wcj = X_wpj * X_j
    # transform from world to child body frame
    X_wc = X_wcj * wp.transform_inverse(X_cj)

    # compute transform of center of mass
    X_cm = body_X_com[child]
    X_sm = X_wc * X_cm

    # store geometry transforms
    body_q[child] = X_wc
    body_q_com[child] = X_sm


@wp.kernel
def eval_rigid_fk(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    body_X_com: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]

    for i in range(start, end):
        compute_link_transform(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_q,
            joint_X_p,
            joint_X_c,
            body_X_com,
            joint_axis,
            joint_axis_start,
            joint_axis_dim,
            body_q,
            body_q_com,
        )


@wp.func
def spatial_cross(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_top(a)
    v_a = wp.spatial_bottom(a)

    w_b = wp.spatial_top(b)
    v_b = wp.spatial_bottom(b)

    w = wp.cross(w_a, w_b)
    v = wp.cross(w_a, v_b) + wp.cross(v_a, w_b)

    return wp.spatial_vector(w, v)


@wp.func
def spatial_cross_dual(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_top(a)
    v_a = wp.spatial_bottom(a)

    w_b = wp.spatial_top(b)
    v_b = wp.spatial_bottom(b)

    w = wp.cross(w_a, w_b) + wp.cross(v_a, v_b)
    v = wp.cross(w_a, v_b)

    return wp.spatial_vector(w, v)


@wp.func
def dense_index(stride: int, i: int, j: int):
    return i * stride + j


@wp.func
def compute_link_velocity(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    gravity: wp.vec3,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    type = joint_type[i]
    child = joint_child[i]
    parent = joint_parent[i]
    q_start = joint_q_start[i]
    qd_start = joint_qd_start[i]

    X_pj = joint_X_p[i]
    # X_cj = joint_X_c[i]

    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj

    # compute motion subspace and velocity across the joint (also stores S_s to global memory)
    axis_start = joint_axis_start[i]
    lin_axis_count = joint_axis_dim[i, 0]
    ang_axis_count = joint_axis_dim[i, 1]
    v_j_s = jcalc_motion(
        type,
        joint_axis,
        axis_start,
        lin_axis_count,
        ang_axis_count,
        X_wpj,
        joint_q,
        joint_qd,
        q_start,
        qd_start,
        joint_S_s,
    )

    # parent velocity
    v_parent_s = wp.spatial_vector()
    a_parent_s = wp.spatial_vector()

    if parent >= 0:
        v_parent_s = body_v_s[parent]
        a_parent_s = body_a_s[parent]

    # body velocity, acceleration
    v_s = v_parent_s + v_j_s
    a_s = a_parent_s + spatial_cross(v_s, v_j_s)  # + joint_S_s[i]*self.joint_qdd[i]

    # compute body forces
    X_sm = body_q_com[child]
    I_m = body_I_m[child]

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    m = I_m[3, 3]

    f_g = m * gravity
    r_com = wp.transform_get_translation(X_sm)
    f_g_s = wp.spatial_vector(wp.cross(r_com, f_g), f_g)

    # body forces
    I_s = spatial_transform_inertia(X_sm, I_m)

    f_b_s = I_s * a_s + spatial_cross_dual(v_s, I_s * v_s)

    body_v_s[child] = v_s
    body_a_s[child] = a_s
    body_f_s[child] = f_b_s - f_g_s
    body_I_s[child] = I_s


# Inverse dynamics via Recursive Newton-Euler algorithm (Featherstone Table 5.1)
@wp.kernel
def eval_rigid_id(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    gravity: wp.vec3,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]

    # compute link velocities and coriolis forces
    for i in range(start, end):
        compute_link_velocity(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_qd,
            joint_axis,
            joint_axis_start,
            joint_axis_dim,
            body_I_m,
            body_q,
            body_q_com,
            joint_X_p,
            joint_X_c,
            gravity,
            joint_S_s,
            body_I_s,
            body_v_s,
            body_f_s,
            body_a_s,
        )


@wp.kernel
def eval_rigid_tau(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    joint_axis_mode: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_act: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_fb_s: wp.array(dtype=wp.spatial_vector),
    body_f_ext: wp.array(dtype=wp.spatial_vector),
    # outputs
    body_ft_s: wp.array(dtype=wp.spatial_vector),
    tau: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]
    count = end - start

    # compute joint forces
    for offset in range(count):
        # for backwards traversal
        i = end - offset - 1

        type = joint_type[i]
        parent = joint_parent[i]
        child = joint_child[i]
        dof_start = joint_qd_start[i]
        coord_start = joint_q_start[i]
        axis_start = joint_axis_start[i]
        lin_axis_count = joint_axis_dim[i, 0]
        ang_axis_count = joint_axis_dim[i, 1]

        # total forces on body
        f_b_s = body_fb_s[child]
        f_t_s = body_ft_s[child]
        f_ext = body_f_ext[child]
        f_s = f_b_s + f_t_s + f_ext

        # compute joint-space forces, writes out tau
        jcalc_tau(
            type,
            joint_target_ke,
            joint_target_kd,
            joint_limit_ke,
            joint_limit_kd,
            joint_S_s,
            joint_q,
            joint_qd,
            joint_act,
            joint_axis_mode,
            joint_limit_lower,
            joint_limit_upper,
            coord_start,
            dof_start,
            axis_start,
            lin_axis_count,
            ang_axis_count,
            f_s,
            tau,
        )

        # update parent forces, todo: check that this is valid for the backwards pass
        if parent >= 0:
            wp.atomic_add(body_ft_s, parent, f_s)


# builds spatial Jacobian J which is an (joint_count*6)x(dof_count) matrix
@wp.kernel
def eval_rigid_jacobian(
    articulation_start: wp.array(dtype=int),
    articulation_J_start: wp.array(dtype=int),
    joint_ancestor: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    # outputs
    J: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    joint_start = articulation_start[index]
    joint_end = articulation_start[index + 1]
    joint_count = joint_end - joint_start

    J_offset = articulation_J_start[index]

    articulation_dof_start = joint_qd_start[joint_start]
    articulation_dof_end = joint_qd_start[joint_end]
    articulation_dof_count = articulation_dof_end - articulation_dof_start

    for i in range(joint_count):
        row_start = i * 6

        j = joint_start + i
        while j != -1:
            joint_dof_start = joint_qd_start[j]
            joint_dof_end = joint_qd_start[j + 1]
            joint_dof_count = joint_dof_end - joint_dof_start

            # fill out each row of the Jacobian walking up the tree
            for dof in range(joint_dof_count):
                col = (joint_dof_start - articulation_dof_start) + dof
                S = joint_S_s[joint_dof_start + dof]

                for k in range(6):
                    J[J_offset + dense_index(articulation_dof_count, row_start + k, col)] = S[k]

            j = joint_ancestor[j]


@wp.func
def spatial_mass(
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    joint_start: int,
    joint_count: int,
    M_start: int,
    # outputs
    M: wp.array(dtype=float),
):
    stride = joint_count * 6
    for l in range(joint_count):
        I = body_I_s[joint_start + l]
        for i in range(6):
            for j in range(6):
                M[M_start + dense_index(stride, l * 6 + i, l * 6 + j)] = I[i, j]


@wp.kernel
def eval_rigid_mass(
    articulation_start: wp.array(dtype=int),
    articulation_M_start: wp.array(dtype=int),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    # outputs
    M: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    joint_start = articulation_start[index]
    joint_end = articulation_start[index + 1]
    joint_count = joint_end - joint_start

    M_offset = articulation_M_start[index]

    spatial_mass(body_I_s, joint_start, joint_count, M_offset, M)


@wp.func
def dense_gemm(
    m: int,
    n: int,
    p: int,
    transpose_A: bool,
    transpose_B: bool,
    add_to_C: bool,
    A_start: int,
    B_start: int,
    C_start: int,
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    # outputs
    C: wp.array(dtype=float),
):
    # multiply a `m x p` matrix A by a `p x n` matrix B to produce a `m x n` matrix C
    for i in range(m):
        for j in range(n):
            sum = float(0.0)
            for k in range(p):
                if transpose_A:
                    a_i = k * m + i
                else:
                    a_i = i * p + k
                if transpose_B:
                    b_j = j * p + k
                else:
                    b_j = k * n + j
                sum += A[A_start + a_i] * B[B_start + b_j]

            if add_to_C:
                C[C_start + i * n + j] += sum
            else:
                C[C_start + i * n + j] = sum


# @wp.func_grad(dense_gemm)
# def adj_dense_gemm(
#     m: int,
#     n: int,
#     p: int,
#     transpose_A: bool,
#     transpose_B: bool,
#     add_to_C: bool,
#     A_start: int,
#     B_start: int,
#     C_start: int,
#     A: wp.array(dtype=float),
#     B: wp.array(dtype=float),
#     # outputs
#     C: wp.array(dtype=float),
# ):
#     add_to_C = True
#     if transpose_A:
#         dense_gemm(p, m, n, False, True, add_to_C, A_start, B_start, C_start, B, wp.adjoint[C], wp.adjoint[A])
#         dense_gemm(p, n, m, False, False, add_to_C, A_start, B_start, C_start, A, wp.adjoint[C], wp.adjoint[B])
#     else:
#         dense_gemm(
#             m, p, n, False, not transpose_B, add_to_C, A_start, B_start, C_start, wp.adjoint[C], B, wp.adjoint[A]
#         )
#         dense_gemm(p, n, m, True, False, add_to_C, A_start, B_start, C_start, A, wp.adjoint[C], wp.adjoint[B])


def create_inertia_matrix_kernel(num_joints, num_dofs):
    @wp.kernel
    def eval_dense_gemm_tile(
        J_arr: wp.array3d(dtype=float), M_arr: wp.array3d(dtype=float), H_arr: wp.array3d(dtype=float)
    ):
        articulation = wp.tid()

        J = wp.tile_load(J_arr[articulation], shape=(wp.static(6 * num_joints), num_dofs))
        P = wp.tile_zeros(shape=(wp.static(6 * num_joints), num_dofs), dtype=float)

        # compute P = M*J where M is a 6x6 block diagonal mass matrix
        for i in range(int(num_joints)):
            # 6x6 block matrices are on the diagonal
            M_body = wp.tile_load(M_arr[articulation], shape=(6, 6), offset=(i * 6, i * 6))

            # load a 6xN row from the Jacobian
            J_body = wp.tile_view(J, offset=(i * 6, 0), shape=(6, num_dofs))

            # compute weighted row
            P_body = wp.tile_matmul(M_body, J_body)

            # assign to the P slice
            wp.tile_assign(P, P_body, offset=(i * 6, 0))

        # compute H = J^T*P
        H = wp.tile_matmul(wp.tile_transpose(J), P)

        wp.tile_store(H_arr[articulation], H)

    return eval_dense_gemm_tile


def create_batched_cholesky_kernel(num_dofs):
    assert num_dofs == 18

    @wp.kernel
    def eval_tiled_dense_cholesky_batched(
        A: wp.array3d(dtype=float), R: wp.array2d(dtype=float), L: wp.array3d(dtype=float)
    ):
        articulation = wp.tid()

        a = wp.tile_load(A[articulation], shape=(num_dofs, num_dofs), storage="shared")
        r = wp.tile_load(R[articulation], shape=num_dofs, storage="shared")
        a_r = wp.tile_diag_add(a, r)
        l = wp.tile_cholesky(a_r)
        wp.tile_store(L[articulation], wp.tile_transpose(l))

    return eval_tiled_dense_cholesky_batched


def create_inertia_matrix_cholesky_kernel(num_joints, num_dofs):
    @wp.kernel
    def eval_dense_gemm_and_cholesky_tile(
        J_arr: wp.array3d(dtype=float),
        M_arr: wp.array3d(dtype=float),
        R_arr: wp.array2d(dtype=float),
        H_arr: wp.array3d(dtype=float),
        L_arr: wp.array3d(dtype=float),
    ):
        articulation = wp.tid()

        J = wp.tile_load(J_arr[articulation], shape=(wp.static(6 * num_joints), num_dofs))
        P = wp.tile_zeros(shape=(wp.static(6 * num_joints), num_dofs), dtype=float)

        # compute P = M*J where M is a 6x6 block diagonal mass matrix
        for i in range(int(num_joints)):
            # 6x6 block matrices are on the diagonal
            M_body = wp.tile_load(M_arr[articulation], shape=(6, 6), offset=(i * 6, i * 6))

            # load a 6xN row from the Jacobian
            J_body = wp.tile_view(J, offset=(i * 6, 0), shape=(6, num_dofs))

            # compute weighted row
            P_body = wp.tile_matmul(M_body, J_body)

            # assign to the P slice
            wp.tile_assign(P, P_body, offset=(i * 6, 0))

        # compute H = J^T*P
        H = wp.tile_matmul(wp.tile_transpose(J), P)
        wp.tile_store(H_arr[articulation], H)

        # cholesky L L^T = (H + diag(R))
        R = wp.tile_load(R_arr[articulation], shape=num_dofs, storage="shared")
        H_R = wp.tile_diag_add(H, R)
        L = wp.tile_cholesky(H_R)
        wp.tile_store(L_arr[articulation], L)

    return eval_dense_gemm_and_cholesky_tile


@wp.kernel
def eval_dense_gemm_batched(
    m: wp.array(dtype=int),
    n: wp.array(dtype=int),
    p: wp.array(dtype=int),
    transpose_A: bool,
    transpose_B: bool,
    A_start: wp.array(dtype=int),
    B_start: wp.array(dtype=int),
    C_start: wp.array(dtype=int),
    A: wp.array(dtype=float),
    B: wp.array(dtype=float),
    C: wp.array(dtype=float),
):
    # on the CPU each thread computes the whole matrix multiply
    # on the GPU each block computes the multiply with one output per-thread
    batch = wp.tid()  # /kNumThreadsPerBlock;
    add_to_C = False

    dense_gemm(
        m[batch],
        n[batch],
        p[batch],
        transpose_A,
        transpose_B,
        add_to_C,
        A_start[batch],
        B_start[batch],
        C_start[batch],
        A,
        B,
        C,
    )


@wp.func
def dense_cholesky(
    n: int,
    A: wp.array(dtype=float),
    R: wp.array(dtype=float),
    A_start: int,
    R_start: int,
    # outputs
    L: wp.array(dtype=float),
):
    # compute the Cholesky factorization of A = L L^T with diagonal regularization R
    for j in range(n):
        s = A[A_start + dense_index(n, j, j)] + R[R_start + j]

        for k in range(j):
            r = L[A_start + dense_index(n, j, k)]
            s -= r * r

        s = wp.sqrt(s)
        invS = 1.0 / s

        L[A_start + dense_index(n, j, j)] = s

        for i in range(j + 1, n):
            s = A[A_start + dense_index(n, i, j)]

            for k in range(j):
                s -= L[A_start + dense_index(n, i, k)] * L[A_start + dense_index(n, j, k)]

            L[A_start + dense_index(n, i, j)] = s * invS


@wp.func_grad(dense_cholesky)
def adj_dense_cholesky(
    n: int,
    A: wp.array(dtype=float),
    R: wp.array(dtype=float),
    A_start: int,
    R_start: int,
    # outputs
    L: wp.array(dtype=float),
):
    # nop, use dense_solve to differentiate through (A^-1)b = x
    pass


@wp.kernel
def eval_dense_cholesky_batched(
    A_starts: wp.array(dtype=int),
    A_dim: wp.array(dtype=int),
    A: wp.array(dtype=float),
    R: wp.array(dtype=float),
    L: wp.array(dtype=float),
):
    batch = wp.tid()

    n = A_dim[batch]
    A_start = A_starts[batch]
    R_start = n * batch

    dense_cholesky(n, A, R, A_start, R_start, L)


@wp.func
def dense_subs(
    n: int,
    L_start: int,
    b_start: int,
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    # outputs
    x: wp.array(dtype=float),
):
    # Solves (L L^T) x = b for x given the Cholesky factor L
    # forward substitution solves the lower triangular system L y = b for y
    for i in range(n):
        s = b[b_start + i]

        for j in range(i):
            s -= L[L_start + dense_index(n, i, j)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + dense_index(n, i, i)]

    # backward substitution solves the upper triangular system L^T x = y for x
    for i in range(n - 1, -1, -1):
        s = x[b_start + i]

        for j in range(i + 1, n):
            s -= L[L_start + dense_index(n, j, i)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + dense_index(n, i, i)]


@wp.func
def dense_solve(
    n: int,
    L_start: int,
    b_start: int,
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    # outputs
    x: wp.array(dtype=float),
    tmp: wp.array(dtype=float),
):
    # helper function to include tmp argument for backward pass
    dense_subs(n, L_start, b_start, L, b, x)


@wp.func_grad(dense_solve)
def adj_dense_solve(
    n: int,
    L_start: int,
    b_start: int,
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    # outputs
    x: wp.array(dtype=float),
    tmp: wp.array(dtype=float),
):
    if not tmp or not wp.adjoint[x] or not wp.adjoint[A] or not wp.adjoint[L]:
        return
    for i in range(n):
        tmp[b_start + i] = 0.0

    dense_subs(n, L_start, b_start, L, wp.adjoint[x], tmp)

    for i in range(n):
        wp.adjoint[b][b_start + i] += tmp[b_start + i]

    # A* = -adj_b*x^T
    for i in range(n):
        for j in range(n):
            wp.adjoint[L][L_start + dense_index(n, i, j)] += -tmp[b_start + i] * x[b_start + j]

    for i in range(n):
        for j in range(n):
            wp.adjoint[A][L_start + dense_index(n, i, j)] += -tmp[b_start + i] * x[b_start + j]


@wp.kernel
def eval_dense_solve_batched(
    L_start: wp.array(dtype=int),
    L_dim: wp.array(dtype=int),
    b_start: wp.array(dtype=int),
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    # outputs
    x: wp.array(dtype=float),
    tmp: wp.array(dtype=float),
):
    batch = wp.tid()

    dense_solve(L_dim[batch], L_start[batch], b_start[batch], A, L, b, x, tmp)


@wp.kernel
def integrate_generalized_joints(
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    type = joint_type[index]
    coord_start = joint_q_start[index]
    dof_start = joint_qd_start[index]
    lin_axis_count = joint_axis_dim[index, 0]
    ang_axis_count = joint_axis_dim[index, 1]

    jcalc_integrate(
        type,
        joint_q,
        joint_qd,
        joint_qdd,
        coord_start,
        dof_start,
        lin_axis_count,
        ang_axis_count,
        dt,
        joint_q_new,
        joint_qd_new,
    )

################################## MOREAU SPECIFIC KERNELS BEGIN ##################################
def matmul_batched(batch_count, m, n, k, t1, t2, A_start, B_start, C_start, A, B, C, device):
    if device == "cpu":
        threads = batch_count
    else:
        threads = 256 * batch_count  # must match the threadblock size used in adjoint.py

    wp.launch(
        kernel=eval_dense_gemm_batched,
        dim=threads,
        inputs=[m, n, k, t1, t2, A_start, B_start, C_start, A, B],
        outputs=[C],
        device=device,
    )

@wp.func
def offset_sigmoid(x: float, scale: float, offset: float):
    return 1.0 / (
        1.0 + wp.exp(wp.clamp(x * scale - offset, -100.0, 50.0))
    )  # clamp for stability (exp gradients) unstable from around 85

@wp.func
def prox_loop_soft(
    tid: int,
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec_0: wp.vec3,
    c_vec_1: wp.vec3,
    c_vec_2: wp.vec3,
    c_vec_3: wp.vec3,
    c_0: float,
    c_1: float,
    c_2: float,
    c_3: float,
    scale: float,
    mu: float,
    prox_iter: int,
    p_0: wp.vec3,
    p_1: wp.vec3,
    p_2: wp.vec3,
    p_3: wp.vec3,
):
    # solve percussions iteratively
    for it in range(prox_iter):
        # CONTACT 0
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 0, 0] * p_0
        r_sum += wp.determinant(G_mat[tid, 0, 0])
        sum += G_mat[tid, 0, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 0, 1])
        sum += G_mat[tid, 0, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 0, 2])
        sum += G_mat[tid, 0, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 0, 3])

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_0 = p_0 - r * (sum + c_vec_0)

        # projection to friction cone
        if p_0[1] <= 0.0:
            p_0 = wp.vec3(0.0, 0.0, 0.0)
        elif p_0[0] != 0.0 or p_0[2] != 0.0:
            fm = wp.sqrt(p_0[0] ** 2.0 + p_0[2] ** 2.0)  # friction magnitude
            if mu * p_0[1] < fm:
                p_0 = wp.vec3(p_0[0] * mu * p_0[1] / fm, p_0[1], p_0[2] * mu * p_0[1] / fm)

        # CONTACT 1
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 1, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 1, 0])
        sum += G_mat[tid, 1, 1] * p_1
        r_sum += wp.determinant(G_mat[tid, 1, 1])
        sum += G_mat[tid, 1, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 1, 2])
        sum += G_mat[tid, 1, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 1, 3])

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_1 = p_1 - r * (sum + c_vec_1)

        # projection to friction cone
        if p_1[1] <= 0.0:
            p_1 = wp.vec3(0.0, 0.0, 0.0)
        elif p_1[0] != 0.0 or p_1[2] != 0.0:
            fm = wp.sqrt(p_1[0] ** 2.0 + p_1[2] ** 2.0)  # friction magnitude
            if mu * p_1[1] < fm:
                p_1 = wp.vec3(p_1[0] * mu * p_1[1] / fm, p_1[1], p_1[2] * mu * p_1[1] / fm)

        # CONTACT 2
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 2, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 2, 0])
        sum += G_mat[tid, 2, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 2, 1])
        sum += G_mat[tid, 2, 2] * p_2
        r_sum += wp.determinant(G_mat[tid, 2, 2])
        sum += G_mat[tid, 2, 3] * p_3 * offset_sigmoid(c_3, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 2, 3])

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_2 = p_2 - r * (sum + c_vec_2)

        # projection to friction cone
        if p_2[1] <= 0.0:
            p_2 = wp.vec3(0.0, 0.0, 0.0)
        elif p_2[0] != 0.0 or p_2[2] != 0.0:
            fm = wp.sqrt(p_2[0] ** 2.0 + p_2[2] ** 2.0)  # friction magnitude
            if mu * p_2[1] < fm:
                p_2 = wp.vec3(p_2[0] * mu * p_2[1] / fm, p_2[1], p_2[2] * mu * p_2[1] / fm)

        # CONTACT 3
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 3, 0] * p_0 * offset_sigmoid(c_0, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 3, 0])
        sum += G_mat[tid, 3, 1] * p_1 * offset_sigmoid(c_1, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 3, 1])
        sum += G_mat[tid, 3, 2] * p_2 * offset_sigmoid(c_2, scale, 0.0)
        r_sum += wp.determinant(G_mat[tid, 3, 2])
        sum += G_mat[tid, 3, 3] * p_3
        r_sum += wp.determinant(G_mat[tid, 3, 3])

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_3 = p_3 - r * (sum + c_vec_3)

        # projection to friction cone
        if p_3[1] <= 0.0:
            p_3 = wp.vec3(0.0, 0.0, 0.0)
        elif p_3[0] != 0.0 or p_3[2] != 0.0:
            fm = wp.sqrt(p_3[0] ** 2.0 + p_3[2] ** 2.0)  # friction magnitude
            if mu * p_3[1] < fm:
                p_3 = wp.vec3(p_3[0] * mu * p_3[1] / fm, p_3[1], p_3[2] * mu * p_3[1] / fm)

    return p_0, p_1, p_2, p_3

@wp.func
def prox_loop(
    tid: int,
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec_0: wp.vec3,
    c_vec_1: wp.vec3,
    c_vec_2: wp.vec3,
    c_vec_3: wp.vec3,
    mu: float,
    prox_iter: int,
    p_0: wp.vec3,
    p_1: wp.vec3,
    p_2: wp.vec3,
    p_3: wp.vec3,
):
    for it in range(prox_iter):
        # CONTACT 0
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 0, 0] * p_0
        r_sum += wp.determinant(G_mat[tid, 0, 0])
        sum += G_mat[tid, 0, 1] * p_1
        r_sum += wp.determinant(G_mat[tid, 0, 1])
        sum += G_mat[tid, 0, 2] * p_2
        r_sum += wp.determinant(G_mat[tid, 0, 2])
        sum += G_mat[tid, 0, 3] * p_3
        r_sum += wp.determinant(G_mat[tid, 0, 3])

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_0 = p_0 - r * (sum + c_vec_0)

        # projection to friction cone
        if p_0[1] <= 0.0:
            p_0 = wp.vec3(0.0, 0.0, 0.0)
        elif p_0[0] != 0.0 or p_0[2] != 0.0:
            fm = wp.sqrt(p_0[0] ** 2.0 + p_0[2] ** 2.0)  # friction magnitude
            if mu * p_0[1] < fm:
                p_0 = wp.vec3(p_0[0] * mu * p_0[1] / fm, p_0[1], p_0[2] * mu * p_0[1] / fm)

        # CONTACT 1
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 1, 0] * p_0
        r_sum += wp.determinant(G_mat[tid, 1, 0])
        sum += G_mat[tid, 1, 1] * p_1
        r_sum += wp.determinant(G_mat[tid, 1, 1])
        sum += G_mat[tid, 1, 2] * p_2
        r_sum += wp.determinant(G_mat[tid, 1, 2])
        sum += G_mat[tid, 1, 3] * p_3
        r_sum += wp.determinant(G_mat[tid, 1, 3])

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_1 = p_1 - r * (sum + c_vec_1)

        # projection to friction cone
        if p_1[1] <= 0.0:
            p_1 = wp.vec3(0.0, 0.0, 0.0)
        elif p_1[0] != 0.0 or p_1[2] != 0.0:
            fm = wp.sqrt(p_1[0] ** 2.0 + p_1[2] ** 2.0)  # friction magnitude
            if mu * p_1[1] < fm:
                p_1 = wp.vec3(p_1[0] * mu * p_1[1] / fm, p_1[1], p_1[2] * mu * p_1[1] / fm)

        # CONTACT 2
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 2, 0] * p_0
        r_sum += wp.determinant(G_mat[tid, 2, 0])
        sum += G_mat[tid, 2, 1] * p_1
        r_sum += wp.determinant(G_mat[tid, 2, 1])
        sum += G_mat[tid, 2, 2] * p_2
        r_sum += wp.determinant(G_mat[tid, 2, 2])
        sum += G_mat[tid, 2, 3] * p_3
        r_sum += wp.determinant(G_mat[tid, 2, 3])

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_2 = p_2 - r * (sum + c_vec_2)

        # projection to friction cone
        if p_2[1] <= 0.0:
            p_2 = wp.vec3(0.0, 0.0, 0.0)
        elif p_2[0] != 0.0 or p_2[2] != 0.0:
            fm = wp.sqrt(p_2[0] ** 2.0 + p_2[2] ** 2.0)  # friction magnitude
            if mu * p_2[1] < fm:
                p_2 = wp.vec3(p_2[0] * mu * p_2[1] / fm, p_2[1], p_2[2] * mu * p_2[1] / fm)

        # CONTACT 3
        # calculate sum(G_ij*p_j) and sum over det(G_ij)
        sum = wp.vec3(0.0, 0.0, 0.0)
        r_sum = 0.0

        sum += G_mat[tid, 3, 0] * p_0
        r_sum += wp.determinant(G_mat[tid, 3, 0])
        sum += G_mat[tid, 3, 1] * p_1
        r_sum += wp.determinant(G_mat[tid, 3, 1])
        sum += G_mat[tid, 3, 2] * p_2
        r_sum += wp.determinant(G_mat[tid, 3, 2])
        sum += G_mat[tid, 3, 3] * p_3
        r_sum += wp.determinant(G_mat[tid, 3, 3])

        r = 1.0 / (1.0 + r_sum)  # +1 for stability

        # update percussion
        p_3 = p_3 - r * (sum + c_vec_3)

        # projection to friction cone
        if p_3[1] <= 0.0:
            p_3 = wp.vec3(0.0, 0.0, 0.0)
        elif p_3[0] != 0.0 or p_3[2] != 0.0:
            fm = wp.sqrt(p_3[0] ** 2.0 + p_3[2] ** 2.0)  # friction magnitude
            if mu * p_3[1] < fm:
                p_3 = wp.vec3(p_3[0] * mu * p_3[1] / fm, p_3[1], p_3[2] * mu * p_3[1] / fm)

    return p_0, p_1, p_2, p_3

@wp.kernel
def construct_contact_jacobian(
    J: wp.array(dtype=float),
    J_start: wp.array(dtype=int),
    Jc_start: wp.array(dtype=int),
    body_X_sc: wp.array(dtype=wp.transform),
    rigid_contact_max: int,
    articulation_count: int,
    dof_count: int,
    contact_body: wp.array(dtype=int),
    contact_point: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    geo: ModelShapeGeometry,
    col_height: float,
    Jc: wp.array(dtype=float),
    c_body_vec: wp.array(dtype=int),
    point_vec: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    contacts_per_articulation = rigid_contact_max / articulation_count

    for i in range(2, contacts_per_articulation):  # iterate (almost) all contacts
        contact_id = tid * contacts_per_articulation + i
        c_body = contact_body[contact_id]
        c_point = contact_point[contact_id]
        c_shape = contact_shape[contact_id]
        c_dist = geo.thickness[c_shape]

        if (c_body - tid) % 3 == 0 and i % 2 == 0:  # only consider foot contacts
            foot_id = (c_body - tid - tid * 12) / 3 - 1
            X_s = body_X_sc[c_body]
            n = wp.vec3(0.0, 1.0, 0.0)
            # transform point to world space
            p = (
                wp.transform_point(X_s, c_point) - n * c_dist
            )  # add on 'thickness' of shape, e.g.: radius of sphere/capsule
            p_skew = wp.skew(wp.vec3(p[0], p[1], p[2]))
            # check ground contact
            c = wp.dot(n, p)

            if c <= col_height:
                # Jc = J_p - skew(p)*J_r
                for j in range(0, 3):  # iterate all contact dofs
                    for k in range(0, dof_count):  # iterate all joint dofs
                        Jc[dense_J_index(Jc_start, 3, dof_count, tid, foot_id, j, k)] = (
                            J[
                                dense_J_index(J_start, 6, dof_count, 0, c_body, j + 3, k)
                            ]  # tid is 0 because c_body already iterates over full J
                            - p_skew[j, 0] * J[dense_J_index(J_start, 6, dof_count, 0, c_body, 0, k)]
                            - p_skew[j, 1] * J[dense_J_index(J_start, 6, dof_count, 0, c_body, 1, k)]
                            - p_skew[j, 2] * J[dense_J_index(J_start, 6, dof_count, 0, c_body, 2, k)]
                        )

            c_body_vec[tid * 4 + foot_id] = c_body
            point_vec[tid * 4 + foot_id] = p

@wp.kernel
def convert_c_to_vector(c: wp.array(dtype=float), c_vec: wp.array2d(dtype=wp.vec3)):
    tid = wp.tid()

    for i in range(4):
        c_start = tid * 3 * 4 + i * 3  # each articulation has 4 contacts, each contact has 3 dimensions
        c_vec[tid, i] = wp.vec3(c[c_start], c[c_start + 1], c[c_start + 2])

@wp.kernel
def eval_dense_add_batched(
    n: wp.array(dtype=int),
    start: wp.array(dtype=int),
    a: wp.array(dtype=float),
    b: wp.array(dtype=float),
    dt: float,
    c: wp.array(dtype=float),
):
    tid = wp.tid()
    for i in range(0, n[tid]):
        c[start[tid] + i] = a[start[tid] + i] + b[start[tid] + i] * dt

@wp.kernel
def split_matrix(
    A: wp.array(dtype=float),
    dof_count: int,
    A_start: wp.array(dtype=int),
    a_start: wp.array(dtype=int),
    a_1: wp.array(dtype=float),
    a_2: wp.array(dtype=float),
    a_3: wp.array(dtype=float),
    a_4: wp.array(dtype=float),
    a_5: wp.array(dtype=float),
    a_6: wp.array(dtype=float),
    a_7: wp.array(dtype=float),
    a_8: wp.array(dtype=float),
    a_9: wp.array(dtype=float),
    a_10: wp.array(dtype=float),
    a_11: wp.array(dtype=float),
    a_12: wp.array(dtype=float),
):
    tid = wp.tid()

    for i in range(dof_count):
        a_1[a_start[tid] + i] = A[A_start[tid] + i]
        a_2[a_start[tid] + i] = A[A_start[tid] + i + 18]
        a_3[a_start[tid] + i] = A[A_start[tid] + i + 36]
        a_4[a_start[tid] + i] = A[A_start[tid] + i + 54]
        a_5[a_start[tid] + i] = A[A_start[tid] + i + 72]
        a_6[a_start[tid] + i] = A[A_start[tid] + i + 90]
        a_7[a_start[tid] + i] = A[A_start[tid] + i + 108]
        a_8[a_start[tid] + i] = A[A_start[tid] + i + 126]
        a_9[a_start[tid] + i] = A[A_start[tid] + i + 144]
        a_10[a_start[tid] + i] = A[A_start[tid] + i + 162]
        a_11[a_start[tid] + i] = A[A_start[tid] + i + 180]
        a_12[a_start[tid] + i] = A[A_start[tid] + i + 198]


@wp.kernel
def create_matrix(
    dof_count: int,
    A_start: wp.array(dtype=int),
    a_start: wp.array(dtype=int),
    a_1: wp.array(dtype=float),
    a_2: wp.array(dtype=float),
    a_3: wp.array(dtype=float),
    a_4: wp.array(dtype=float),
    a_5: wp.array(dtype=float),
    a_6: wp.array(dtype=float),
    a_7: wp.array(dtype=float),
    a_8: wp.array(dtype=float),
    a_9: wp.array(dtype=float),
    a_10: wp.array(dtype=float),
    a_11: wp.array(dtype=float),
    a_12: wp.array(dtype=float),
    A: wp.array(dtype=float),
):
    tid = wp.tid()

    for i in range(dof_count):
        A[A_start[tid] + i] = a_1[a_start[tid] + i]
        A[A_start[tid] + i + 18] = a_2[a_start[tid] + i]
        A[A_start[tid] + i + 36] = a_3[a_start[tid] + i]
        A[A_start[tid] + i + 54] = a_4[a_start[tid] + i]
        A[A_start[tid] + i + 72] = a_5[a_start[tid] + i]
        A[A_start[tid] + i + 90] = a_6[a_start[tid] + i]
        A[A_start[tid] + i + 108] = a_7[a_start[tid] + i]
        A[A_start[tid] + i + 126] = a_8[a_start[tid] + i]
        A[A_start[tid] + i + 144] = a_9[a_start[tid] + i]
        A[A_start[tid] + i + 162] = a_10[a_start[tid] + i]
        A[A_start[tid] + i + 180] = a_11[a_start[tid] + i]
        A[A_start[tid] + i + 198] = a_12[a_start[tid] + i]

@wp.kernel
def prox_iteration_unrolled_soft(
    point_vec: wp.array(dtype=wp.vec3),
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec: wp.array2d(dtype=wp.vec3),
    mu: float,
    prox_iter: int,
    scale_array: wp.array(dtype=float),
    percussion: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    scale = scale_array[0]
    n = wp.vec3(0.0, 1.0, 0.0)
    point_0 = point_vec[tid * 4]
    point_1 = point_vec[tid * 4 + 1]
    point_2 = point_vec[tid * 4 + 2]
    point_3 = point_vec[tid * 4 + 3]
    c_0 = wp.dot(n, point_0)
    c_1 = wp.dot(n, point_1)
    c_2 = wp.dot(n, point_2)
    c_3 = wp.dot(n, point_3)
    c_vec_0 = c_vec[tid, 0]  # * offset_sigmoid(c_0, scale, 0.0)
    c_vec_1 = c_vec[tid, 1]  # * offset_sigmoid(c_1, scale, 0.0)
    c_vec_2 = c_vec[tid, 2]  # * offset_sigmoid(c_2, scale, 0.0)
    c_vec_3 = c_vec[tid, 3]  # * offset_sigmoid(c_3, scale, 0.0)

    # initialize percussions with steady state
    p_0 = -wp.inverse(G_mat[tid, 0, 0]) * c_vec_0
    p_1 = -wp.inverse(G_mat[tid, 1, 1]) * c_vec_1
    p_2 = -wp.inverse(G_mat[tid, 2, 2]) * c_vec_2
    p_3 = -wp.inverse(G_mat[tid, 3, 3]) * c_vec_3

    p_0, p_1, p_2, p_3 = prox_loop_soft(
        tid, G_mat, c_vec_0, c_vec_1, c_vec_2, c_vec_3, c_0, c_1, c_2, c_3, scale, mu, prox_iter, p_0, p_1, p_2, p_3
    )

    percussion[tid, 0] = p_0 * offset_sigmoid(c_0, scale, 0.0)
    percussion[tid, 1] = p_1 * offset_sigmoid(c_1, scale, 0.0)
    percussion[tid, 2] = p_2 * offset_sigmoid(c_2, scale, 0.0)
    percussion[tid, 3] = p_3 * offset_sigmoid(c_3, scale, 0.0)

@wp.kernel
def prox_iteration_unrolled(
    G_mat: wp.array3d(dtype=wp.mat33),
    c_vec: wp.array2d(dtype=wp.vec3),
    mu: float,
    prox_iter: int,
    percussion: wp.array2d(dtype=wp.vec3),
):
    tid = wp.tid()

    c_vec_0 = c_vec[tid, 0]
    c_vec_1 = c_vec[tid, 1]
    c_vec_2 = c_vec[tid, 2]
    c_vec_3 = c_vec[tid, 3]

    # initialize percussions with steady state
    p_0 = -wp.inverse(G_mat[tid, 0, 0]) * c_vec_0
    p_1 = -wp.inverse(G_mat[tid, 1, 1]) * c_vec_1
    p_2 = -wp.inverse(G_mat[tid, 2, 2]) * c_vec_2
    p_3 = -wp.inverse(G_mat[tid, 3, 3]) * c_vec_3
    # overwrite percussions with steady state only in normal direction
    # p_0 = wp.vec3(0.0, p_0[1], 0.0)
    # p_1 = wp.vec3(0.0, p_1[1], 0.0)
    # p_2 = wp.vec3(0.0, p_2[1], 0.0)
    # p_3 = wp.vec3(0.0, p_3[1], 0.0)

    p_0, p_1, p_2, p_3 = prox_loop(tid, G_mat, c_vec_0, c_vec_1, c_vec_2, c_vec_3, mu, prox_iter, p_0, p_1, p_2, p_3)

    percussion[tid, 0] = p_0
    percussion[tid, 1] = p_1
    percussion[tid, 2] = p_2
    percussion[tid, 3] = p_3

@wp.kernel
def p_to_f_s(
    c_body_vec: wp.array(dtype=int),
    point_vec: wp.array(dtype=wp.vec3),
    percussion: wp.array2d(dtype=wp.vec3),
    dt: float,
    body_f_s: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    for i in range(4):
        f = -percussion[tid, i] / dt
        t = wp.cross(point_vec[tid * 4 + i], f)
        wp.atomic_add(body_f_s, c_body_vec[tid * 4 + i], wp.spatial_vector(t, f))

@wp.func
def dense_J_index(J_start: wp.array(dtype=int), dim_count: int, dof_count: int, tid: int, i: int, j: int, k: int):
    """
    J_start: articulation start index
    dim_count: number of body/contact dims
    dof_count: number of joint dofs

    tid: articulation
    i: body/contact
    j: linear/angular velocity
    k: joint velocity
    """

    return J_start[tid] + i * dim_count * dof_count + j * dof_count + k  # articulation, body/contact, dim, dof

@wp.func
def dense_G_index(G_start: wp.array(dtype=int), tid: int, i: int, j: int, k: int, l: int):
    """
    Calculates flat index for G stored in row-major order.
    tid: articulation index
    i: block row index (contact 1, 0..3)
    j: block col index (contact 2, 0..3)
    k: row index within 3x3 block (0..2)
    l: col index within 3x3 block (0..2)
    """
    # Assuming N=4 contacts per articulation (hardcoded in loops using G_mat)
    num_contacts = 4
    num_block_cols = num_contacts  # G is (N*3) x (N*3)
    num_total_cols = num_block_cols * 3  # Total number of columns in the flat matrix per articulation

    global_row = i * 3 + k
    global_col = j * 3 + l

    return G_start[tid] + global_row * num_total_cols + global_col

@wp.kernel
def convert_G_to_matrix(G_start: wp.array(dtype=int), G: wp.array(dtype=float), G_mat: wp.array3d(dtype=wp.mat33)):
    tid = wp.tid()

    for i in range(4):
        for j in range(4):
            G_mat[tid, i, j] = wp.mat33(
                G[dense_G_index(G_start, tid, i, j, 0, 0)],
                G[dense_G_index(G_start, tid, i, j, 0, 1)],
                G[dense_G_index(G_start, tid, i, j, 0, 2)],
                G[dense_G_index(G_start, tid, i, j, 1, 0)],
                G[dense_G_index(G_start, tid, i, j, 1, 1)],
                G[dense_G_index(G_start, tid, i, j, 1, 2)],
                G[dense_G_index(G_start, tid, i, j, 2, 0)],
                G[dense_G_index(G_start, tid, i, j, 2, 1)],
                G[dense_G_index(G_start, tid, i, j, 2, 2)],
            )

@wp.kernel 
def map_shape_contacts_to_body_contacts(
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int), 
    shape_body: wp.array(dtype=int),
    contact_body0: wp.array(dtype=int),
    contact_body1: wp.array(dtype=int)
):
    i = wp.tid()
    contact_body0[i] = shape_body[contact_shape0[i]]
    contact_body1[i] = shape_body[contact_shape1[i]]

################################## MOREAU SPECIFIC KERNELS  END  ##################################


class MoreauIntegrator(Integrator):
    """A semi-implicit integrator using symplectic Euler that operates
    on reduced (also called generalized) coordinates to simulate articulated rigid body dynamics
    based on Featherstone's composite rigid body algorithm (CRBA).

    See: Featherstone, Roy. Rigid Body Dynamics Algorithms. Springer US, 2014.

    Instead of maximal coordinates :attr:`State.body_q` (rigid body positions) and :attr:`State.body_qd`
    (rigid body velocities) as is the case :class:`SemiImplicitIntegrator`, :class:`FeatherstoneIntegrator`
    uses :attr:`State.joint_q` and :attr:`State.joint_qd` to represent the positions and velocities of
    joints without allowing any redundant degrees of freedom.

    After constructing :class:`Model` and :class:`State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Note:
        Unlike :class:`SemiImplicitIntegrator` and :class:`XPBDIntegrator`, :class:`FeatherstoneIntegrator` does not simulate rigid bodies with nonzero mass as floating bodies if they are not connected through any joints. Floating-base systems require an explicit free joint with which the body is connected to the world, see :meth:`ModelBuilder.add_joint_free`.

    Semi-implicit time integration is a variational integrator that
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    Example
    -------

    .. code-block:: python

        integrator = wp.FeatherstoneIntegrator(model)

        # simulation loop
        for i in range(100):
            state = integrator.simulate(model, state_in, state_out, dt)

    Note:
        The :class:`FeatherstoneIntegrator` requires the :class:`Model` to be passed in as a constructor argument.

    """

    def __init__(
        self,
        model,
        angular_damping=0.05,
        update_mass_matrix_every=1,
        friction_smoothing=1.0,
        use_tile_gemm=False,
        fuse_cholesky=True,
    ):
        """
        Args:
            model (Model): the model to be simulated.
            angular_damping (float, optional): Angular damping factor. Defaults to 0.05.
            update_mass_matrix_every (int, optional): How often to update the mass matrix (every n-th time the :meth:`simulate` function gets called). Defaults to 1.
            friction_smoothing (float, optional): The delta value for the Huber norm (see :func:`warp.math.norm_huber`) used for the friction velocity normalization. Defaults to 1.0.
        """
        self.angular_damping = angular_damping
        self.update_mass_matrix_every = update_mass_matrix_every
        self.friction_smoothing = friction_smoothing
        self.use_tile_gemm = use_tile_gemm
        self.fuse_cholesky = fuse_cholesky

        self._step = 0

        self.compute_articulation_indices(model)
        self.allocate_model_aux_vars(model)

        if self.use_tile_gemm:
            # create a custom kernel to evaluate the system matrix for this type
            if self.fuse_cholesky:
                self.eval_inertia_matrix_cholesky_kernel = create_inertia_matrix_cholesky_kernel(
                    int(self.joint_count), int(self.dof_count)
                )
            else:
                self.eval_inertia_matrix_kernel = create_inertia_matrix_kernel(
                    int(self.joint_count), int(self.dof_count)
                )

            # ensure matrix is reloaded since otherwise an unload can happen during graph capture
            # todo: should not be necessary?
            wp.load_module(device=wp.get_device())

    def compute_articulation_indices(self, model):
        # calculate total size and offsets of Jacobian and mass matrices for entire system
        if model.joint_count:
            self.J_size = 0
            self.M_size = 0
            self.H_size = 0
            # Moreau specific additions
            self.Jc_size = 0
            self.Jc_row_size = 0
            self.G_size = 0

            articulation_J_start = []
            articulation_M_start = []
            articulation_H_start = []
            # Moreau specific additions
            articulation_Jc_start = []
            articulation_Jc_row_start = []
            articulation_G_start = []

            articulation_M_rows = []
            articulation_H_rows = []
            articulation_J_rows = []
            articulation_J_cols = []
            # Moreau specific additions
            articulation_Jc_rows = []
            articulation_Jc_cols = []
            articulation_G_rows = []
            articulation_vec_size = []

            articulation_dof_start = []
            articulation_coord_start = []
            # Moreau specific additions
            articulation_contact_dim_start = []
            first_contact_dim = 0

            articulation_start = model.articulation_start.numpy()
            joint_q_start = model.joint_q_start.numpy()
            joint_qd_start = model.joint_qd_start.numpy()

            for i in range(model.articulation_count):
                first_joint = articulation_start[i]
                last_joint = articulation_start[i + 1]

                first_coord = joint_q_start[first_joint]

                first_dof = joint_qd_start[first_joint]
                last_dof = joint_qd_start[last_joint]

                joint_count = last_joint - first_joint
                dof_count = last_dof - first_dof

                articulation_J_start.append(self.J_size)
                articulation_M_start.append(self.M_size)
                articulation_H_start.append(self.H_size)
                articulation_dof_start.append(first_dof)
                articulation_coord_start.append(first_coord)
                # Moreau specific additions
                articulation_Jc_start.append(self.Jc_size)
                for i in range(4*3):
                    articulation_Jc_row_start.append(self.Jc_row_size)
                    self.Jc_row_size += dof_count
                articulation_G_start.append(self.G_size)
                articulation_contact_dim_start.append(first_contact_dim)

                # bit of data duplication here, but will leave it as such for clarity
                articulation_M_rows.append(joint_count * 6)
                articulation_H_rows.append(dof_count)
                articulation_J_rows.append(joint_count * 6)
                articulation_J_cols.append(dof_count)
                # Moreau specific additions
                articulation_Jc_rows.append(4*3)
                articulation_Jc_cols.append(dof_count)
                articulation_G_rows.append(4*3)
                articulation_vec_size.append(1)

                if self.use_tile_gemm:
                    # store the joint and dof count assuming all
                    # articulations have the same structure
                    self.joint_count = joint_count
                    self.dof_count = dof_count

                self.J_size += 6 * joint_count * dof_count
                self.M_size += 6 * joint_count * 6 * joint_count
                self.H_size += dof_count * dof_count
                # Moreau specific additions
                self.Jc_size += dof_count*4*3 # assuming 4 contacts per articulation
                self.G_size += 4*3*4*3


            # matrix offsets for batched gemm
            self.articulation_J_start = wp.array(articulation_J_start, dtype=wp.int32, device=model.device)
            self.articulation_M_start = wp.array(articulation_M_start, dtype=wp.int32, device=model.device)
            self.articulation_H_start = wp.array(articulation_H_start, dtype=wp.int32, device=model.device)
            # Moreau specific additions
            self.articulation_H_start_matrix = wp.array([x for x in articulation_H_start for _ in range(4*3)], dtype=wp.int32)
            self.articulation_Jc_start = wp.array(articulation_Jc_start, dtype=wp.int32)
            self.articulation_Jc_row_start = wp.array(articulation_Jc_row_start, dtype=wp.int32)
            self.articulation_G_start = wp.array(articulation_G_start, dtype=wp.int32)


            self.articulation_M_rows = wp.array(articulation_M_rows, dtype=wp.int32, device=model.device)
            self.articulation_H_rows = wp.array(articulation_H_rows, dtype=wp.int32, device=model.device)
            self.articulation_J_rows = wp.array(articulation_J_rows, dtype=wp.int32, device=model.device)
            self.articulation_J_cols = wp.array(articulation_J_cols, dtype=wp.int32, device=model.device)
            # Moreau specific additions
            self.articulation_Jc_rows = wp.array(articulation_Jc_rows, dtype=wp.int32)
            self.articulation_Jc_cols = wp.array(articulation_Jc_cols, dtype=wp.int32)
            self.articulation_G_rows = wp.array(articulation_G_rows, dtype=wp.int32)
            self.articulation_vec_size = wp.array(articulation_vec_size, dtype=wp.int32)


            self.articulation_dof_start = wp.array(articulation_dof_start, dtype=wp.int32, device=model.device)
            self.articulation_coord_start = wp.array(articulation_coord_start, dtype=wp.int32, device=model.device)
            # Moreau specific additions
            self.articulation_contact_dim_start = wp.array(articulation_contact_dim_start, dtype=wp.int32)

    def allocate_model_aux_vars(self, model):
        # allocate mass, Jacobian matrices, and other auxiliary variables pertaining to the model
        if model.joint_count:

            ################### MOREAU SPECIFIC ALLOCATIONS BEGIN ###################
            #self.sigmoid_scale = wp.zeros(1, dtype=wp.float32, requires_grad=requires_grad)
            self.sigmoid_scale = wp.array([1.0], dtype=wp.float32)

            # Contact Jacobian matrix
            self.Jc = wp.zeros((self.Jc_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad)
            # self.Jc = wp.zeros(self.Jc_size, dtype=wp.float32, requires_grad=True)
            
            # Delassus matrix (flattened)  
            self.G = wp.zeros((self.G_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad)
            # self.G = wp.zeros(self.G_size, dtype=wp.float32, requires_grad=True)
            
            # Delassus matrix (matrix form)
            self.G_mat = wp.zeros((model.articulation_count, 4, 4), dtype=wp.mat33, device=model.device, requires_grad=model.requires_grad)
            # self.G_mat = wp.zeros((self.articulation_count,4,4), dtype=wp.mat33, requires_grad=True)
            
            # Contact body vectors
            self.c_body_vec = wp.zeros((model.articulation_count*4,), dtype=wp.int32, device=model.device)
            # self.c_body_vec = wp.zeros(self.articulation_count*4, dtype=wp.int32, device=self.device)

            self.col_height = 0.0

            # Create body contact arrays
            self.rigid_contact_body0 = wp.empty_like(model.rigid_contact_shape0)
            self.rigid_contact_body1 = wp.empty_like(model.rigid_contact_shape1)

            
            ################### MOREAU SPECIFIC ALLOCATIONS  END  ###################

            # system matrices
            self.M = wp.zeros((self.M_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad)
            self.J = wp.zeros((self.J_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad)
            self.P = wp.empty_like(self.J, requires_grad=model.requires_grad)
            self.H = wp.empty((self.H_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad)

            # zero since only upper triangle is set which can trigger NaN detection
            self.L = wp.zeros_like(self.H)

        if model.body_count:
            self.body_I_m = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=model.device, requires_grad=model.requires_grad
            )
            wp.launch(
                compute_spatial_inertia,
                model.body_count,
                inputs=[model.body_inertia, model.body_mass],
                outputs=[self.body_I_m],
                device=model.device,
            )
            self.body_X_com = wp.empty(
                (model.body_count,), dtype=wp.transform, device=model.device, requires_grad=model.requires_grad
            )
            wp.launch(
                compute_com_transforms,
                model.body_count,
                inputs=[model.body_com],
                outputs=[self.body_X_com],
                device=model.device,
            )

    def allocate_state_aux_vars(self, model, target, requires_grad):
        # allocate auxiliary variables that vary with state
        if model.body_count:
            # joints
            target.joint_qdd = wp.zeros_like(model.joint_qd, requires_grad=requires_grad)
            target.joint_tau = wp.empty_like(model.joint_qd, requires_grad=requires_grad)
            if requires_grad:
                # used in the custom grad implementation of eval_dense_solve_batched
                target.joint_solve_tmp = wp.zeros_like(model.joint_qd, requires_grad=True)
            else:
                target.joint_solve_tmp = None
            target.joint_S_s = wp.empty(
                (model.joint_dof_count,),
                dtype=wp.spatial_vector,
                device=model.device,
                requires_grad=requires_grad,
            )

            # derived rigid body data (maximal coordinates)
            target.body_q_com = wp.empty_like(model.body_q, requires_grad=requires_grad)
            target.body_I_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=model.device, requires_grad=requires_grad
            )
            target.body_v_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_a_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_f_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_ft_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            ################### MOREAU SPECIFIC ALLOCATIONS BEGIN ###################
            # target.body_X_sc = wp.zeros(
            #                 (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            #             )
            
            target.body_X_sc = wp.zeros((model.body_count), dtype=wp.transformf, requires_grad=True)
            
            target.point_vec = wp.zeros(model.articulation_count*4, dtype=wp.vec3, requires_grad=True)
            target.percussion = wp.zeros((model.articulation_count, 4), dtype=wp.vec3, requires_grad=True)

            # compute G and c
            target.inv_m_times_h = wp.zeros_like(model.joint_qd, requires_grad=True) # maybe set to 0?
            target.Jc_times_inv_m_times_h = wp.zeros((model.articulation_count*4*3,), requires_grad=True)
            target.Jc_qd = wp.zeros((model.articulation_count*4*3,), requires_grad=True)
            target.c = wp.zeros((model.articulation_count*4*3,), requires_grad=True)
            target.c_vec = wp.zeros((model.articulation_count, 4), dtype=wp.vec3, requires_grad=True)
            # s.JcT_p = wp.zeros_like(self.joint_qd, requires_grad=True)
            target.tmp_inv_m_times_h = wp.zeros_like(model.joint_qd, requires_grad=True)

            target.Jc_1 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_2 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_3 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_4 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_5 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_6 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_7 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_8 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_9 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_10 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_11 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Jc_12 = wp.zeros_like(model.joint_qd, requires_grad=True)

            target.Inv_M_times_Jc_t_1 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_2 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_3 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_4 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_5 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_6 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_7 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_8 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_9 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_10 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_11 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.Inv_M_times_Jc_t_12 = wp.zeros_like(model.joint_qd, requires_grad=True)

            target.Inv_M_times_Jc_t = wp.zeros((self.Jc_size,), dtype=wp.float32, requires_grad=True)
            
            target.tmp_1 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_2 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_3 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_4 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_5 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_6 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_7 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_8 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_9 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_10 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_11 = wp.zeros_like(model.joint_qd, requires_grad=True)
            target.tmp_12 = wp.zeros_like(model.joint_qd, requires_grad=True)
            ################### MOREAU SPECIFIC ALLOCATIONS  END  ###################

            target._featherstone_augmented = True

    def simulate(self, model: Model, state_in: State, state_out: State, dt: float, control: Control = None):
        requires_grad = state_in.requires_grad

        # optionally create dynamical auxiliary variables
        if requires_grad:
            state_aug = state_out
        else:
            state_aug = self

        if not getattr(state_aug, "_featherstone_augmented", False):
            self.allocate_state_aux_vars(model, state_aug, requires_grad)
        if control is None:
            control = model.control(clone_variables=False)

        # EXPLICIT clearing for Moreau Specific State additions
        if model.articulation_count:
            # Zero contact forces that accumulate
            state_aug.body_f_s.zero_()           # Spatial contact forces
            
            if hasattr(state_aug, 'percussion'):
                state_aug.percussion.zero_()      # Contact impulses
                
            # Zero contact computation intermediates (safer to clear these too)
            if hasattr(state_aug, 'c_vec'):
                state_aug.c_vec.zero_()
            if hasattr(state_aug, 'c'):
                state_aug.c.zero_()
            if hasattr(state_aug, 'Jc_qd'):
                state_aug.Jc_qd.zero_()
            if hasattr(state_aug, 'Jc_times_inv_m_times_h'):
                state_aug.Jc_times_inv_m_times_h.zero_()

        with wp.ScopedTimer("simulate", False):
            particle_f = None
            body_f = None

            if state_in.particle_count:
                particle_f = state_in.particle_f

            if state_in.body_count:
                body_f = state_in.body_f

            # damped springs
            eval_spring_forces(model, state_in, particle_f)

            # triangle elastic and lift/drag forces
            eval_triangle_forces(model, state_in, control, particle_f)

            # triangle/triangle contacts
            eval_triangle_contact_forces(model, state_in, particle_f)

            # triangle bending
            eval_bending_forces(model, state_in, particle_f)

            # tetrahedral FEM
            eval_tetrahedral_forces(model, state_in, control, particle_f)

            # particle-particle interactions
            eval_particle_forces(model, state_in, particle_f)

            # particle ground contacts
            eval_particle_ground_contact_forces(model, state_in, particle_f)

            # particle shape contact
            eval_particle_body_contact_forces(model, state_in, particle_f, body_f, body_f_in_world_frame=True)

            # muscles
            if False:
                eval_muscle_forces(model, state_in, control, body_f)

            # ----------------------------
            # articulations

            if model.joint_count:
                # evaluate body transforms
                wp.launch(
                    eval_rigid_fk,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_q_start,
                        state_in.joint_q,
                        model.joint_X_p,
                        model.joint_X_c,
                        self.body_X_com,
                        model.joint_axis,
                        model.joint_axis_start,
                        model.joint_axis_dim,
                    ],
                    outputs=[state_in.body_q, state_aug.body_q_com],
                    device=model.device,
                )

                # print("body_X_sc:")
                # print(state_in.body_q.numpy())

                # evaluate joint inertias, motion vectors, and forces
                state_aug.body_f_s.zero_()
                wp.launch(
                    eval_rigid_id,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        model.joint_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_in.joint_q,
                        state_in.joint_qd,
                        model.joint_axis,
                        model.joint_axis_start,
                        model.joint_axis_dim,
                        self.body_I_m,
                        state_in.body_q,
                        state_aug.body_q_com,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.gravity,
                    ],
                    outputs=[
                        state_aug.joint_S_s,
                        state_aug.body_I_s,
                        state_aug.body_v_s,
                        state_aug.body_f_s,
                        state_aug.body_a_s,
                    ],
                    device=model.device,
                )

                ##################### MOREAU SPECIFIC CONTACT RESOLUTION BEGIN #####################
                
                
                if model.articulation_count:

                    if True:
                        # Map: body = shape_body[shape]
                        wp.launch(
                            kernel=map_shape_contacts_to_body_contacts,
                            dim=model.rigid_contact_max,
                            inputs=[model.rigid_contact_shape0, model.rigid_contact_shape1, model.shape_body],
                            outputs=[self.rigid_contact_body0, self.rigid_contact_body1]
                        )

                    # eval_tau (tau will be h)
                    # evaluate joint torques
                    state_aug.body_ft_s.zero_()
                    wp.launch(
                        eval_rigid_tau,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_start,
                            model.joint_type,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_q_start,
                            model.joint_qd_start,
                            model.joint_axis_start,
                            model.joint_axis_dim,
                            model.joint_axis_mode,
                            state_in.joint_q,
                            state_in.joint_qd,
                            control.joint_act,
                            model.joint_target_ke,
                            model.joint_target_kd,
                            model.joint_limit_lower,
                            model.joint_limit_upper,
                            model.joint_limit_ke,
                            model.joint_limit_kd,
                            state_aug.joint_S_s,
                            state_aug.body_f_s,
                            body_f,
                        ],
                        outputs=[
                            state_aug.body_ft_s,
                            state_aug.joint_tau,
                        ],
                        device=model.device,
                    )

                    # eval Jc, G, and c
                    self.eval_contact_quantities(model, state_in, state_aug, dt)

                    # prox iteration
                    # self.eval_contact_forces(model, state_aug, dt, mu, prox_iter, mode) # add inputs to simulate later
                    self.eval_contact_forces(model, state_aug, dt)

                ##################### MOREAU SPECIFIC CONTACT RESOLUTION  END  #####################

                """
                # OLD SOFT CONTACT EVALUATION
                if model.rigid_contact_max and (
                    (model.ground and model.shape_ground_contact_pair_count) or model.shape_contact_pair_count
                ):
                    wp.launch(
                        kernel=eval_rigid_contacts,
                        dim=model.rigid_contact_max,
                        inputs=[
                            state_in.body_q,
                            state_aug.body_v_s,
                            model.body_com,
                            model.shape_materials,
                            model.shape_geo,
                            model.shape_body,
                            model.rigid_contact_count,
                            model.rigid_contact_point0,
                            model.rigid_contact_point1,
                            model.rigid_contact_normal,
                            model.rigid_contact_shape0,
                            model.rigid_contact_shape1,
                            True,
                            self.friction_smoothing,
                        ],
                        outputs=[body_f],
                        device=model.device,
                    )

                    # if model.rigid_contact_count.numpy()[0] > 0:
                    #     print(body_f.numpy())
                """

                if model.articulation_count:
                    # evaluate joint torques
                    state_aug.body_ft_s.zero_()
                    wp.launch(
                        eval_rigid_tau,
                        dim=model.articulation_count,
                        inputs=[
                            model.articulation_start,
                            model.joint_type,
                            model.joint_parent,
                            model.joint_child,
                            model.joint_q_start,
                            model.joint_qd_start,
                            model.joint_axis_start,
                            model.joint_axis_dim,
                            model.joint_axis_mode,
                            state_in.joint_q,
                            state_in.joint_qd,
                            control.joint_act,
                            model.joint_target_ke,
                            model.joint_target_kd,
                            model.joint_limit_lower,
                            model.joint_limit_upper,
                            model.joint_limit_ke,
                            model.joint_limit_kd,
                            state_aug.joint_S_s,
                            state_aug.body_f_s,
                            body_f,
                        ],
                        outputs=[
                            state_aug.body_ft_s,
                            state_aug.joint_tau,
                        ],
                        device=model.device,
                    )

                    # print("joint_tau:")
                    # print(state_aug.joint_tau.numpy())
                    # print("body_q:")
                    # print(state_in.body_q.numpy())
                    # print("body_qd:")
                    # print(state_in.body_qd.numpy())

                    if self._step % self.update_mass_matrix_every == 0:
                        # build J
                        wp.launch(
                            eval_rigid_jacobian,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_start,
                                self.articulation_J_start,
                                model.joint_ancestor,
                                model.joint_qd_start,
                                state_aug.joint_S_s,
                            ],
                            outputs=[self.J],
                            device=model.device,
                        )

                        # build M
                        wp.launch(
                            eval_rigid_mass,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_start,
                                self.articulation_M_start,
                                state_aug.body_I_s,
                            ],
                            outputs=[self.M],
                            device=model.device,
                        )

                        if self.use_tile_gemm:
                            # reshape arrays
                            M_tiled = self.M.reshape((-1, 6 * self.joint_count, 6 * self.joint_count))
                            J_tiled = self.J.reshape((-1, 6 * self.joint_count, self.dof_count))
                            R_tiled = model.joint_armature.reshape((-1, self.dof_count))
                            H_tiled = self.H.reshape((-1, self.dof_count, self.dof_count))
                            L_tiled = self.L.reshape((-1, self.dof_count, self.dof_count))
                            assert H_tiled.shape == (model.articulation_count, 18, 18)
                            assert L_tiled.shape == (model.articulation_count, 18, 18)
                            assert R_tiled.shape == (model.articulation_count, 18)

                            if self.fuse_cholesky:
                                wp.launch_tiled(
                                    self.eval_inertia_matrix_cholesky_kernel,
                                    dim=model.articulation_count,
                                    inputs=[J_tiled, M_tiled, R_tiled],
                                    outputs=[H_tiled, L_tiled],
                                    device=model.device,
                                    block_dim=64,
                                )

                            else:
                                wp.launch_tiled(
                                    self.eval_inertia_matrix_kernel,
                                    dim=model.articulation_count,
                                    inputs=[J_tiled, M_tiled],
                                    outputs=[H_tiled],
                                    device=model.device,
                                    block_dim=256,
                                )

                                wp.launch(
                                    eval_dense_cholesky_batched,
                                    dim=model.articulation_count,
                                    inputs=[
                                        self.articulation_H_start,
                                        self.articulation_H_rows,
                                        self.H,
                                        model.joint_armature,
                                    ],
                                    outputs=[self.L],
                                    device=model.device,
                                )

                            # import numpy as np
                            # J = J_tiled.numpy()
                            # M = M_tiled.numpy()
                            # R = R_tiled.numpy()
                            # for i in range(model.articulation_count):
                            #     r = R[i,:,0]
                            #     H = J[i].T @ M[i] @ J[i]
                            #     L = np.linalg.cholesky(H + np.diag(r))
                            #     np.testing.assert_allclose(H, H_tiled.numpy()[i], rtol=1e-2, atol=1e-2)
                            #     np.testing.assert_allclose(L, L_tiled.numpy()[i], rtol=1e-1, atol=1e-1)

                        else:
                            # form P = M*J
                            wp.launch(
                                eval_dense_gemm_batched,
                                dim=model.articulation_count,
                                inputs=[
                                    self.articulation_M_rows,
                                    self.articulation_J_cols,
                                    self.articulation_J_rows,
                                    False,
                                    False,
                                    self.articulation_M_start,
                                    self.articulation_J_start,
                                    # P start is the same as J start since it has the same dims as J
                                    self.articulation_J_start,
                                    self.M,
                                    self.J,
                                ],
                                outputs=[self.P],
                                device=model.device,
                            )

                            # form H = J^T*P
                            wp.launch(
                                eval_dense_gemm_batched,
                                dim=model.articulation_count,
                                inputs=[
                                    self.articulation_J_cols,
                                    self.articulation_J_cols,
                                    # P rows is the same as J rows
                                    self.articulation_J_rows,
                                    True,
                                    False,
                                    self.articulation_J_start,
                                    # P start is the same as J start since it has the same dims as J
                                    self.articulation_J_start,
                                    self.articulation_H_start,
                                    self.J,
                                    self.P,
                                ],
                                outputs=[self.H],
                                device=model.device,
                            )

                            # compute decomposition
                            wp.launch(
                                eval_dense_cholesky_batched,
                                dim=model.articulation_count,
                                inputs=[
                                    self.articulation_H_start,
                                    self.articulation_H_rows,
                                    self.H,
                                    model.joint_armature,
                                ],
                                outputs=[self.L],
                                device=model.device,
                            )

                        # print("joint_act:")
                        # print(control.joint_act.numpy())
                        # print("joint_tau:")
                        # print(state_aug.joint_tau.numpy())
                        # print("H:")
                        # print(self.H.numpy())
                        # print("L:")
                        # print(self.L.numpy())

                    # solve for qdd
                    state_aug.joint_qdd.zero_()
                    wp.launch(
                        eval_dense_solve_batched,
                        dim=model.articulation_count,
                        inputs=[
                            self.articulation_H_start,
                            self.articulation_H_rows,
                            self.articulation_dof_start,
                            self.H,
                            self.L,
                            state_aug.joint_tau,
                        ],
                        outputs=[
                            state_aug.joint_qdd,
                            state_aug.joint_solve_tmp,
                        ],
                        device=model.device,
                    )
                    # print("joint_qdd:")
                    # print(state_aug.joint_qdd.numpy())
                    # print("\n\n")

            # -------------------------------------
            # integrate bodies

            if model.joint_count:
                wp.launch(
                    kernel=integrate_generalized_joints,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_axis_dim,
                        state_in.joint_q,
                        state_in.joint_qd,
                        state_aug.joint_qdd,
                        dt,
                    ],
                    outputs=[state_out.joint_q, state_out.joint_qd],
                    device=model.device,
                )

                # update maximal coordinates
                eval_fk(model, state_out.joint_q, state_out.joint_qd, None, state_out)

            self.integrate_particles(model, state_in, state_out, dt)

            self._step += 1

            return state_out
    
    ########################### MOREAU SPECIFIC METHODS BELOW ###########################
    def eval_contact_quantities(self, model, state_in, state_mid, dt):
        # construct J_c
        # kernel 16
        wp.launch(
            kernel=construct_contact_jacobian,
            dim=model.articulation_count,
            inputs=[
                self.J,
                self.articulation_J_start,
                self.articulation_Jc_start,
                state_mid.body_X_sc,
                model.rigid_contact_max,
                model.articulation_count,
                int(model.joint_dof_count / model.articulation_count),
                self.rigid_contact_body0,
                model.rigid_contact_point0,
                model.rigid_contact_shape0,
                model.shape_geo,
                self.col_height,
            ],
            outputs=[self.Jc, self.c_body_vec, state_mid.point_vec],
            device=model.device,
        )

        # solve for X^T (X = H^-1*Jc^T)
        wp.launch(
            kernel=split_matrix,
            dim=model.articulation_count,
            inputs=[
                self.Jc,
                int(model.joint_dof_count / model.articulation_count),
                self.articulation_Jc_start,
                self.articulation_dof_start,
            ],
            outputs=[
                state_mid.Jc_1,
                state_mid.Jc_2,
                state_mid.Jc_3,
                state_mid.Jc_4,
                state_mid.Jc_5,
                state_mid.Jc_6,
                state_mid.Jc_7,
                state_mid.Jc_8,
                state_mid.Jc_9,
                state_mid.Jc_10,
                state_mid.Jc_11,
                state_mid.Jc_12,
            ],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.H,
                self.L,
                state_mid.Jc_1,
                state_mid.tmp_1,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_1],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.H,
                self.L,
                state_mid.Jc_2,
                state_mid.tmp_2,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_2],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.H,
                self.L,
                state_mid.Jc_3,
                state_mid.tmp_3,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_3],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.H,
                self.L,
                state_mid.Jc_4,
                state_mid.tmp_4,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_4],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.H,
                self.L,
                state_mid.Jc_5,
                state_mid.tmp_5,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_5],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.H,
                self.L,
                state_mid.Jc_6,
                state_mid.tmp_6,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_6],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.H,
                self.L,
                state_mid.Jc_7,
                state_mid.tmp_7,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_7],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.H,
                self.L,
                state_mid.Jc_8,
                state_mid.tmp_8,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_8],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.H,
                self.L,
                state_mid.Jc_9,
                state_mid.tmp_9,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_9],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.H,
                self.L,
                state_mid.Jc_10,
                state_mid.tmp_10,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_10],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.H,
                self.L,
                state_mid.Jc_11,
                state_mid.tmp_11,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_11],
            device=model.device,
        )

        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.H,
                self.L,
                state_mid.Jc_12,
                state_mid.tmp_12,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t_12],
            device=model.device,
        )

        wp.launch(
            kernel=create_matrix,
            dim=model.articulation_count,
            inputs=[
                int(model.joint_dof_count / model.articulation_count),
                self.articulation_Jc_start,
                self.articulation_dof_start,
                state_mid.Inv_M_times_Jc_t_1,
                state_mid.Inv_M_times_Jc_t_2,
                state_mid.Inv_M_times_Jc_t_3,
                state_mid.Inv_M_times_Jc_t_4,
                state_mid.Inv_M_times_Jc_t_5,
                state_mid.Inv_M_times_Jc_t_6,
                state_mid.Inv_M_times_Jc_t_7,
                state_mid.Inv_M_times_Jc_t_8,
                state_mid.Inv_M_times_Jc_t_9,
                state_mid.Inv_M_times_Jc_t_10,
                state_mid.Inv_M_times_Jc_t_11,
                state_mid.Inv_M_times_Jc_t_12,
            ],
            outputs=[state_mid.Inv_M_times_Jc_t],
        )

        # compute G = Jc*(H^-1*Jc^T)
        # kernel 14
        matmul_batched(
            model.articulation_count,
            self.articulation_Jc_rows,  # m
            self.articulation_Jc_rows,  # n
            self.articulation_Jc_cols,  # intermediate dim
            0,
            1,
            self.articulation_Jc_start,
            self.articulation_Jc_start,
            self.articulation_G_start,
            self.Jc,
            state_mid.Inv_M_times_Jc_t,
            self.G,
            device=model.device,
        )

        # convert G to matrix
        # kernel 13
        wp.launch(
            kernel=convert_G_to_matrix,
            dim=model.articulation_count,
            inputs=[self.articulation_G_start, self.G],
            outputs=[self.G_mat],
            device=model.device,
        )

        # solve for x (x = H^-1*h(tau))
        # kernel 12
        wp.launch(
            kernel=eval_dense_solve_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.H,
                self.L,
                state_mid.joint_tau,
                state_mid.tmp_inv_m_times_h,
            ],
            outputs=[state_mid.inv_m_times_h],
            device=model.device,
        )

        # compute Jc*(H^-1*h(tau))
        # kernel 11
        matmul_batched(
            model.articulation_count,
            self.articulation_Jc_rows,  # m
            self.articulation_vec_size,  # n
            self.articulation_Jc_cols,  # intermediate dim
            0,
            0,
            self.articulation_Jc_start,
            self.articulation_dof_start,
            self.articulation_contact_dim_start,
            self.Jc,
            state_mid.inv_m_times_h,
            state_mid.Jc_times_inv_m_times_h,
            device=model.device,
        )

        # compute Jc*qd
        # kernel 10
        matmul_batched(
            model.articulation_count,
            self.articulation_Jc_rows,  # m
            self.articulation_vec_size,  # n
            self.articulation_Jc_cols,  # intermediate dim
            0,
            0,
            self.articulation_Jc_start,
            self.articulation_dof_start,
            self.articulation_contact_dim_start,
            self.Jc,
            state_in.joint_qd,
            state_mid.Jc_qd,
            device=model.device,
        )

        # compute Jc*qd + Jc*(H^-1*h(tau)) * dt
        # kernel 9
        wp.launch(
            kernel=eval_dense_add_batched,
            dim=model.articulation_count,
            inputs=[
                self.articulation_Jc_rows,
                self.articulation_contact_dim_start,
                state_mid.Jc_qd,
                state_mid.Jc_times_inv_m_times_h,
                dt,
            ],
            outputs=[state_mid.c],
            device=model.device,
        )

        # convert c to matrix/vector arrays
        # kernel 8
        wp.launch(
            kernel=convert_c_to_vector,
            dim=model.articulation_count,
            inputs=[state_mid.c],
            outputs=[state_mid.c_vec],
            device=model.device,
        )

    def eval_contact_forces(self, model, state_mid, dt, mu = 0.0, prox_iter = 50, mode = "soft"):
        # prox iteration
        # kernel 7
        if mode == "hard":
            wp.launch(
                kernel=prox_iteration_unrolled,
                dim=model.articulation_count,
                inputs=[self.G_mat, state_mid.c_vec, mu, prox_iter],
                outputs=[state_mid.percussion],
                device=model.device,
            )
        elif mode == "soft":
            wp.launch(
                kernel=prox_iteration_unrolled_soft,
                dim=model.articulation_count,
                inputs=[state_mid.point_vec, self.G_mat, state_mid.c_vec, mu, prox_iter, self.sigmoid_scale], # before model.sigmoid_scale
                outputs=[state_mid.percussion],
                device=model.device,
            )
        else:
            raise ValueError("Invalid mode")

        # kernel 6
        wp.launch(
            kernel=p_to_f_s,
            dim=model.articulation_count,
            inputs=[self.c_body_vec, state_mid.point_vec, state_mid.percussion, dt],
            outputs=[state_mid.body_f_s],
            device=model.device,
        )