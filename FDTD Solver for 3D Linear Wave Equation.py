"""
FDTD Solver for 3D Linear Wave Equation - GPU/CPU Performance Comparison Version
Equation: ∂²p/∂t² = c² ∇²p
"""

import numpy as np
import time
import warnings
import sys

# Detect if running in Jupyter/Colab environment
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        IN_JUPYTER = True
        print("Detected Jupyter/Colab environment")
    else:
        IN_JUPYTER = False
except ImportError:
    IN_JUPYTER = False

# Set matplotlib backend
import matplotlib
if IN_JUPYTER:
    # Use inline backend in Jupyter
    matplotlib.use('module://matplotlib_inline.backend_inline')
else:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font settings
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
rcParams['axes.unicode_minus'] = False

print(f"NumPy version: {np.__version__}")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy available, will use GPU acceleration")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not installed, GPU unavailable")

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE and not CUPY_AVAILABLE:
        print("PyTorch available (CUDA), will use GPU acceleration")
except ImportError:
    TORCH_AVAILABLE = False


class FDTD3D:

    # 3D FDTD Wave Equation Solver
    def __init__(self, Lx, Ly, Lz, dx, c=343, dt=None, use_gpu=True):
        # Set compute device
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np

        # Grid dimensions
        self.Nx = int(Lx / dx) + 1
        self.Ny = int(Ly / dx) + 1
        self.Nz = int(Lz / dx) + 1
        self.dx = dx
        self.c = c

        # Courant number (Under the seven-point stencil of the Cartesian grid, a = b = 0)
        self.lambda_cfl = 1.0 / np.sqrt(3)

        # Time steps (dt < lambda * dx / c)
        if dt is None:
            self.dt = self.lambda_cfl * dx / c
        else:
            self.dt = dt
            lambda_actual = c * self.dt / dx
            if lambda_actual > self.lambda_cfl:
                print(f" Warning: Courant number {lambda_actual:.3f} > {self.lambda_cfl:.3f}")

        self.lambda2 = (c * self.dt / dx) ** 2
        self.n_points = self.Nx * self.Ny * self.Nz

        self.coeff_self = 2 - 6 * self.lambda2
        self.coeff_neighbor = self.lambda2

        self.allocate_memory()

        # Neumann boundary condition
        self.boundary_type = 'rigid'
        self.step_count = 0

    def allocate_memory(self):
        shape = (self.Nx, self.Ny, self.Nz)
        if self.use_gpu:
            self.p_prev = self.xp.zeros(shape, dtype=self.xp.float32)
            self.p_curr = self.xp.zeros(shape, dtype=self.xp.float32)
            self.p_next = self.xp.zeros(shape, dtype=self.xp.float32)
        else:
            self.p_prev = np.zeros(shape, dtype=np.float32, order='C')
            self.p_curr = np.zeros(shape, dtype=np.float32, order='C')
            self.p_next = np.zeros(shape, dtype=np.float32, order='C')

    def to_numpy(self, array):
        if self.use_gpu:
            return self.xp.asnumpy(array)
        return array

    def ricker_source(self, fc, duration):
        t = np.arange(0, duration, self.dt)
        tau = 1.0 / (np.pi * fc)
        t0 = 1.5 * tau
        source = (1 - 2 * np.pi**2 * fc**2 * (t - t0)**2) * np.exp(-np.pi**2 * fc**2 * (t - t0)**2)
        source = source.astype(np.float32)
        if self.use_gpu:
            return self.xp.asarray(source)
        return source

    def gaussian_source(self, fc, duration):
        t = np.arange(0, duration, self.dt)
        sigma = 1.0 / (2 * np.pi * fc)
        source = np.exp(-0.5 * ((t - 3*sigma) / sigma) ** 2)
        source = source.astype(np.float32)
        if self.use_gpu:
            return self.xp.asarray(source)
        return source

    def reset(self):
        if self.use_gpu:
            self.p_prev.fill(0)
            self.p_curr.fill(0)
            self.p_next.fill(0)
        else:
            self.p_prev.fill(0)
            self.p_curr.fill(0)
            self.p_next.fill(0)
        self.step_count = 0

    def step(self):
        """Single timestep update"""
        S = (self.p_curr[2:, 1:-1, 1:-1] +
             self.p_curr[:-2, 1:-1, 1:-1] +
             self.p_curr[1:-1, 2:, 1:-1] +
             self.p_curr[1:-1, :-2, 1:-1] +
             self.p_curr[1:-1, 1:-1, 2:] +
             self.p_curr[1:-1, 1:-1, :-2])

        self.p_next[1:-1, 1:-1, 1:-1] = (self.coeff_self * self.p_curr[1:-1, 1:-1, 1:-1]
                                         + self.coeff_neighbor * S
                                         - self.p_prev[1:-1, 1:-1, 1:-1])

        self.p_curr[0, :, :] = self.p_curr[1, :, :]
        self.p_curr[-1, :, :] = self.p_curr[-2, :, :]
        self.p_curr[:, 0, :] = self.p_curr[:, 1, :]
        self.p_curr[:, -1, :] = self.p_curr[:, -2, :]
        self.p_curr[:, :, 0] = self.p_curr[:, :, 1]
        self.p_curr[:, :, -1] = self.p_curr[:, :, -2]

        self.p_prev, self.p_curr, self.p_next = self.p_curr, self.p_next, self.p_prev
        self.step_count += 1

    def run(self, n_steps, source_position=None, source_waveform=None,
            receiver_positions=None, progress_interval=500, verbose=True):

        receiver_signals = {}
        if receiver_positions is not None:
            for name in receiver_positions:
                receiver_signals[name] = []

        if verbose:
            print(f"\nStarting simulation ({n_steps} time steps)...")
        start_time = time.time()

        for step in range(n_steps):
            if source_position is not None and source_waveform is not None:
                if step < len(source_waveform):
                    ix, iy, iz = source_position
                    self.p_curr[ix, iy, iz] += source_waveform[step]

            self.step()

            if receiver_positions is not None:
                for name, pos in receiver_positions.items():
                    ix, iy, iz = pos
                    val = self.p_curr[ix, iy, iz]
                    if self.use_gpu:
                        val = float(val.item())
                    else:
                        val = float(val)
                    receiver_signals[name].append(val)

            if verbose and (step + 1) % progress_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed
                remaining = (n_steps - step - 1) / steps_per_sec if steps_per_sec > 0 else 0
                print(f"  Step {step+1:6d}/{n_steps} | Speed: {steps_per_sec:.0f} steps/s | Remaining: {remaining:.0f} s")

        elapsed = time.time() - start_time
        if verbose:
            print(f"\n Simulation completed! Total time: {elapsed:.2f} seconds")
            print(f"  Average speed: {n_steps/elapsed:.0f} steps/s")

        return receiver_signals, elapsed

    def get_pressure_field(self):
        return self.to_numpy(self.p_curr)


def plot_comparison_results(t, gpu_signals, cpu_signals, pressure_field_gpu, pressure_field_cpu,
                            Lx, Ly, Lz, gpu_time, cpu_time, speedup, save_path='fdtd_comparison.png'):
    """
    Plot GPU vs CPU comparison results
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. GPU receiver signals
    ax1 = axes[0, 0]
    for name, signal in gpu_signals.items():
        signal_array = np.array(signal)
        ax1.plot(t[:len(signal_array)], signal_array, label=name, alpha=0.7)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Pressure [Pa]')
    ax1.set_title('GPU - Receiver Signals')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. CPU receiver signals
    ax2 = axes[0, 1]
    for name, signal in cpu_signals.items():
        signal_array = np.array(signal)
        ax2.plot(t[:len(signal_array)], signal_array, label=name, alpha=0.7)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Pressure [Pa]')
    ax2.set_title('CPU - Receiver Signals')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Signal comparison (center)
    ax3 = axes[0, 2]
    gpu_center = np.array(gpu_signals.get('center', list(gpu_signals.values())[0]))
    cpu_center = np.array(cpu_signals.get('center', list(cpu_signals.values())[0]))
    ax3.plot(t[:len(gpu_center)], gpu_center, label='GPU', alpha=0.7)
    ax3.plot(t[:len(cpu_center)], cpu_center, label='CPU', alpha=0.7, linestyle='--')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Pressure [Pa]')
    ax3.set_title('Center Signal Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. GPU pressure field slice
    ax4 = axes[1, 0]
    if pressure_field_gpu.ndim == 3:
        slice_idx = pressure_field_gpu.shape[2] // 2
        im1 = ax4.imshow(pressure_field_gpu[:, :, slice_idx].T,
                         extent=[0, Lx, 0, Ly],
                         origin='lower', cmap='RdBu_r')
        ax4.set_xlabel('x [m]')
        ax4.set_ylabel('y [m]')
        ax4.set_title(f'GPU Pressure Field (z = {Lz/2:.1f} m)')
        plt.colorbar(im1, ax=ax4, label='Pressure [Pa]')

    # 5. CPU pressure field slice
    ax5 = axes[1, 1]
    if pressure_field_cpu.ndim == 3:
        slice_idx = pressure_field_cpu.shape[2] // 2
        im2 = ax5.imshow(pressure_field_cpu[:, :, slice_idx].T,
                         extent=[0, Lx, 0, Ly],
                         origin='lower', cmap='RdBu_r')
        ax5.set_xlabel('x [m]')
        ax5.set_ylabel('y [m]')
        ax5.set_title(f'CPU Pressure Field (z = {Lz/2:.1f} m)')
        plt.colorbar(im2, ax=ax5, label='Pressure [Pa]')

    # 6. Performance comparison
    ax6 = axes[1, 2]
    devices = ['GPU', 'CPU']
    times = [gpu_time, cpu_time]
    colors = ['#2ecc71', '#e74c3c']
    bars = ax6.bar(devices, times, color=colors, alpha=0.7)
    ax6.set_ylabel('Time (seconds)')
    ax6.set_title(f'Performance Comparison (Speedup: {speedup:.1f}x)')

    # Add value labels on top of bars
    for bar, time_val in zip(bars, times):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{time_val:.1f}s', ha='center', va='bottom')

    # Add improvement annotation
    improvement = (cpu_time - gpu_time) / cpu_time * 100
    ax6.text(0.5, max(times) * 0.8, f'GPU is {improvement:.1f}% faster',
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison figure saved to: {save_path}")

    if IN_JUPYTER:
        from IPython.display import Image, display
        display(Image(filename=save_path))
    else:
        plt.show()

    plt.close(fig)


def run_performance_comparison():
    """
    Run GPU vs CPU performance comparison
    """
    print("="*70)
    print("GPU vs CPU Performance Comparison Test")
    print("="*70)

    # ========== Test parameters ==========
    # Adjust grid size as needed
    test_configs = [
        {'name': 'Small Grid', 'Lx': 1.0, 'Ly': 1.0, 'Lz': 1.0, 'dx': 0.05, 'n_steps': 500},
        {'name': 'Medium Grid', 'Lx': 2.0, 'Ly': 2.0, 'Lz': 2.0, 'dx': 0.025, 'n_steps': 1000},
        {'name': 'Large Grid', 'Lx': 2.0, 'Ly': 2.0, 'Lz': 2.0, 'dx': 0.02, 'n_steps': 2000},
    ]

    # Select which configuration to run (default: medium grid)
    config = test_configs[1]  # 0: Small, 1: Medium, 2: Large
    # For quick testing, use small grid
    RUN_QUICK_TEST = False  # Set to True to run small grid quick test

    if RUN_QUICK_TEST:
        config = test_configs[0]
        print("\n⚠ Quick test mode (small grid)")

    Lx, Ly, Lz = config['Lx'], config['Ly'], config['Lz']
    dx = config['dx']
    n_steps = config['n_steps']
    c = 343
    fc = 500
    duration = 0.01

    print(f"\nTest configuration: {config['name']}")
    print(f"  Grid dimensions: {Lx}×{Ly}×{Lz} m, dx = {dx*1000:.1f} mm")
    print(f"  Number of time steps: {n_steps}")

    # Create CPU solver (first to get grid information)
    print("\n" + "-"*50)
    print("CPU Mode (NumPy) Test")
    print("-"*50)

    solver_cpu = FDTD3D(Lx, Ly, Lz, dx, c, use_gpu=False)

    # Source position (center)
    source_pos = (solver_cpu.Nx//2, solver_cpu.Ny//2, solver_cpu.Nz//2)
    source_waveform_cpu = solver_cpu.ricker_source(fc, duration)

    # Receiver positions
    receivers = {
        'center': (solver_cpu.Nx//2, solver_cpu.Ny//2, solver_cpu.Nz//2),
        'offset_x': (solver_cpu.Nx//2 + 10, solver_cpu.Ny//2, solver_cpu.Nz//2),
        'offset_y': (solver_cpu.Nx//2, solver_cpu.Ny//2 + 10, solver_cpu.Nz//2),
        'offset_z': (solver_cpu.Nx//2, solver_cpu.Ny//2, solver_cpu.Nz//2 + 10),
    }

    print(f"Source position: {source_pos}")
    print(f"Number of receivers: {len(receivers)}")
    print(f"Grid points: {solver_cpu.n_points:,}")

    # Run CPU simulation
    start_cpu = time.time()
    cpu_signals, cpu_time = solver_cpu.run(
        n_steps=n_steps,
        source_position=source_pos,
        source_waveform=source_waveform_cpu,
        receiver_positions=receivers,
        progress_interval=max(1, n_steps // 4),
        verbose=True
    )

    # Get CPU pressure field
    pressure_field_cpu = solver_cpu.get_pressure_field()

    # Check if GPU is available
    if not CUPY_AVAILABLE:
        print("\n" + "="*50)
        print("GPU unavailable, cannot run comparison test")
        print("Please install CuPy: pip install cupy-cuda12x")
        print("="*50)
        return None, None, None, None, None, None, None

    print("\n" + "-"*50)
    print("GPU Mode (CuPy) Test")
    print("-"*50)

    # Create GPU solver
    solver_gpu = FDTD3D(Lx, Ly, Lz, dx, c, use_gpu=True)

    # Generate source waveform (must be regenerated in GPU mode)
    source_waveform_gpu = solver_gpu.ricker_source(fc, duration)

    # Run GPU simulation
    start_gpu = time.time()
    gpu_signals, gpu_time = solver_gpu.run(
        n_steps=n_steps,
        source_position=source_pos,
        source_waveform=source_waveform_gpu,
        receiver_positions=receivers,
        progress_interval=max(1, n_steps // 4),
        verbose=True
    )

    # Get GPU pressure field
    pressure_field_gpu = solver_gpu.get_pressure_field()

    # ========== Performance comparison statistics ==========
    print("\n" + "="*70)
    print("Performance Comparison Results")
    print("="*70)

    speedup = cpu_time / gpu_time

    print(f"\n{'Device':<10} {'Total Time (s)':<15} {'Steps/s':<15} {'Relative Speed':<15}")
    print("-"*55)
    print(f"{'CPU (NumPy)':<10} {cpu_time:<15.2f} {n_steps/cpu_time:<15.0f} {1.0:<15.1f}x")
    print(f"{'GPU (CuPy)':<10} {gpu_time:<15.2f} {n_steps/gpu_time:<15.0f} {speedup:<15.1f}x")

    print(f"\n GPU Speedup: {speedup:.2f}x")
    print(f" GPU Time Saved: {cpu_time - gpu_time:.2f} seconds")
    print(f" Performance Improvement: {(speedup - 1) * 100:.1f}%")

    # Validate result consistency
    print("\n" + "="*70)
    print("Result Consistency Validation")
    print("="*70)

    # Compare center receiver signals
    gpu_center = np.array(gpu_signals['center'])
    cpu_center = np.array(cpu_signals['center'])

    # Calculate differences
    max_diff = np.max(np.abs(gpu_center - cpu_center))
    mean_diff = np.mean(np.abs(gpu_center - cpu_center))
    rms_diff = np.sqrt(np.mean((gpu_center - cpu_center)**2))
    max_abs_gpu = np.max(np.abs(gpu_center))

    print(f"\nCenter receiver signal comparison:")
    print(f"  Maximum absolute difference: {max_diff:.6e} Pa")
    print(f"  Mean absolute difference: {mean_diff:.6e} Pa")
    print(f"  RMS difference: {rms_diff:.6e} Pa")
    print(f"  Relative difference: {max_diff/max_abs_gpu*100:.4f}%")

    if max_diff < 1e-5:
        print("Results highly consistent, GPU/CPU computation equivalent")
    elif max_diff < 1e-3:
        print("Results generally consistent, small differences due to floating point precision")
    else:
        print("Results differ significantly, please check implementation")

    # Calculate pressure field difference
    field_diff = np.abs(pressure_field_cpu - pressure_field_gpu)
    print(f"\nPressure field difference:")
    print(f"  Maximum difference: {np.max(field_diff):.6e} Pa")
    print(f"  Mean difference: {np.mean(field_diff):.6e} Pa")

    # Prepare visualization data
    t = np.arange(n_steps) * solver_cpu.dt

    # Plot comparison figure
    try:
        plot_comparison_results(
            t, gpu_signals, cpu_signals,
            pressure_field_gpu, pressure_field_cpu,
            Lx, Ly, Lz, gpu_time, cpu_time, speedup
        )
    except Exception as e:
        print(f"Visualization failed: {e}")

    # Save comparison data
    comparison_data = {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'speedup': speedup,
        'n_steps': n_steps,
        'grid_points': solver_cpu.n_points,
        'dx': dx,
        'c': c,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'cpu_signals': {k: np.array(v) for k, v in cpu_signals.items()},
        'gpu_signals': {k: np.array(v) for k, v in gpu_signals.items()},
    }

    np.savez('gpu_cpu_comparison.npz', **comparison_data)
    print(f"\nComparison data saved to: gpu_cpu_comparison.npz")

    return solver_gpu, solver_cpu, gpu_signals, cpu_signals, gpu_time, cpu_time, speedup


def plot_speedup_vs_grid_size():
    """
    Test speedup across different grid sizes
    """
    print("\n" + "="*70)
    print("Speedup Test Across Different Grid Sizes")
    print("="*70)

    if not CUPY_AVAILABLE:
        print("GPU unavailable, cannot run test")
        return None

    # Test different grid sizes
    test_sizes = [
        {'name': '32³', 'Lx': 1.0, 'Ly': 1.0, 'Lz': 1.0, 'dx': 0.0323, 'n_steps': 200},
        {'name': '48³', 'Lx': 1.5, 'Ly': 1.5, 'Lz': 1.5, 'dx': 0.0319, 'n_steps': 300},
        {'name': '64³', 'Lx': 2.0, 'Ly': 2.0, 'Lz': 2.0, 'dx': 0.0317, 'n_steps': 400},
        {'name': '96³', 'Lx': 3.0, 'Ly': 3.0, 'Lz': 3.0, 'dx': 0.0316, 'n_steps': 500},
    ]

    results = []
    c = 343
    fc = 500
    duration = 0.005

    for config in test_sizes:
        print(f"\nTesting: {config['name']} grid")

        Lx, Ly, Lz = config['Lx'], config['Ly'], config['Lz']
        dx = config['dx']
        n_steps = config['n_steps']

        # CPU simulation
        solver_cpu = FDTD3D(Lx, Ly, Lz, dx, c, use_gpu=False)
        source_pos = (solver_cpu.Nx//2, solver_cpu.Ny//2, solver_cpu.Nz//2)
        source_waveform = solver_cpu.ricker_source(fc, duration)
        receivers = {'center': source_pos}

        start = time.time()
        _, cpu_time = solver_cpu.run(n_steps, source_pos, source_waveform, receivers,
                                     progress_interval=n_steps+1, verbose=False)

        # GPU simulation
        solver_gpu = FDTD3D(Lx, Ly, Lz, dx, c, use_gpu=True)
        source_waveform_gpu = solver_gpu.ricker_source(fc, duration)

        start = time.time()
        _, gpu_time = solver_gpu.run(n_steps, source_pos, source_waveform_gpu, receivers,
                                     progress_interval=n_steps+1, verbose=False)

        speedup = cpu_time / gpu_time
        points = solver_cpu.n_points

        results.append({
            'size': config['name'],
            'points': points,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup
        })

        print(f"  Grid points: {points:,}")
        print(f"  CPU: {cpu_time:.2f}s, GPU: {gpu_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

    # Plot speedup curve
    fig, ax = plt.subplots(figsize=(10, 6))

    sizes = [r['size'] for r in results]
    speedups = [r['speedup'] for r in results]
    points = [r['points'] for r in results]

    bars = ax.bar(sizes, speedups, color='#3498db', alpha=0.7)
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Speedup (GPU/CPU)')
    ax.set_title('GPU Speedup vs Grid Size')
    ax.axhline(y=1, color='r', linestyle='--', label='Baseline (CPU)')

    # Add value labels
    for bar, sp, pts in zip(bars, speedups, points):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{sp:.1f}x\n({pts:,} pts)', ha='center', va='bottom', fontsize=9)

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('speedup_analysis.png', dpi=150)
    plt.show()

    # Save results
    np.savez('speedup_results.npz', results=results)
    print(f"\nSpeedup analysis saved to: speedup_analysis.png")

    return results


# ========== Main program ==========
if __name__ == "__main__":
    # Run GPU vs CPU performance comparison
    print("GPU vs CPU Performance Comparison Test")

    # Run main comparison test
    result = run_performance_comparison()

    # Optional: Run speedup analysis across different grid sizes
    RUN_SPEEDUP_ANALYSIS = True  # Set to True to run speedup analysis
    if RUN_SPEEDUP_ANALYSIS and CUPY_AVAILABLE:
        plot_speedup_vs_grid_size()