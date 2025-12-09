import warp as wp

# Minimal test: can gradients flow through eval_rigid_tau at all?
tape = wp.Tape()

# Create test arrays
joint_act = wp.array([1.0], dtype=float, device="cuda", requires_grad=True)
joint_tau = wp.zeros(1, dtype=float, device="cuda", requires_grad=True)

with tape:
    # Simplest possible kernel
    @wp.kernel
    def test_kernel(act: wp.array(dtype=float), tau: wp.array(dtype=float)):
        tau[0] = act[0] * 2.0
    
    wp.launch(test_kernel, dim=1, inputs=[joint_act], outputs=[joint_tau], device="cuda")

# Set output gradient
joint_tau.grad = wp.array([1.0], dtype=float, device="cuda")

tape.backward()
print(f"joint_act gradient: {joint_act.grad.numpy()}")  # Should be [2.0]