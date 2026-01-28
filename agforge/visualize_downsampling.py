
import argparse
import os
import time
import sys
import numpy as np
import trimesh
import torch

# Genesis imports will be conditional/lazy where appropriate or global if safe
import genesis as gs

def mode_generate(args):
    """
    Headless Generation Mode:
    1. Builds simulation.
    2. Steps physics.
    3. Runs downsampling.
    4. Generates meshes.
    5. Saves to disk.
    """
    print("=== MODE: GENERATE (Headless) ===")
    
    # Lazy imports for simulation components to avoid polluting view mode if it shared this process (though we recommend separate processes)
    import gstaichi as ti
    import genesis.utils.particle as pu
    from options import TeleopOptions
    from agforge_builder import build_env
    from reconstruction import SurfaceReconstructor
    from genesis.utils.misc import ti_to_numpy

    print("Building simulation environment...")
    cfg = TeleopOptions()
    cfg.general.show_viewer = False  # Headless
    
    # Build env (initializes Genesis for Simulation backend)
    env = build_env(cfg)
    
    # 1. Warmup / Settle
    print("Stepping simulation to settle particles...")
    for _ in range(5):
        env.scene.step()
    
    # 2. Extract Particles
    print("Extracting particles...")
    solver = env.scene.sim.mpm_solver
    ti.sync()
    
    # Position logic
    if hasattr(solver.particles_render.pos, 'to_numpy'):
         particles = solver.particles_render.pos.to_numpy()[:, 0]
    else:
         particles = ti_to_numpy(solver.particles_render.pos)[:, 0]
    
    # Apply Offset
    offset = env.scene.envs_offset[0]
    if hasattr(offset, 'cpu'): offset = offset.cpu().numpy()
    elif hasattr(offset, 'numpy'): offset = offset.numpy()
    particles = particles + offset
    
    # Filter Active
    if hasattr(solver.particles_render.active, 'to_numpy'):
        active = solver.particles_render.active.to_numpy()[:, 0].astype(bool)
    else:
        active = ti_to_numpy(solver.particles_render.active)[:, 0].astype(bool)
    particles = particles[active]
    
    total_count = len(particles)
    print(f"Total Active Particles: {total_count}")
    
    if total_count == 0:
        print("Error: No active particles found.")
        return

    # 3. Downsampling (The Algorithm to Validate)
    fraction = 0.5
    target_count = int(total_count * fraction)
    print(f"Running CVT Downsampling... Target: {target_count} ({fraction*100}%)")
    
    reconstructor = SurfaceReconstructor(env)
    
    # Manually call the internal sampler for validation
    t0 = time.time()
    particles_tensor = torch.from_numpy(particles).float().to(env.device)
    indices = reconstructor._compute_optimal_indices_torch(particles_tensor, target_count)
    dt_sample = time.time() - t0
    print(f"Sampling completed in {dt_sample:.4f}s")
    
    indices_np = indices.cpu().numpy()
    particles_sampled = particles[indices_np]
    
    # 4. Mesh Generation (Splashsurf)
    output_dir = "data/debug_meshes"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Reconstructing FULL mesh (Reference)...")
    mesh_full = pu.particles_to_mesh(
        positions=particles,
        radius=solver.particle_radius,
        backend='splashsurf'
    )
    
    print("Reconstructing SAMPLED mesh (Test)...")
    # Apply the radius scaling law: R_new = R_old * (1/fraction)^(1/3) * buffer
    radius_scale = (1.0 / fraction) ** (1.0/3.0) * 1.1
    radius_sampled = solver.particle_radius * radius_scale
    print(f"Radius Scaling: {solver.particle_radius:.5f} -> {radius_sampled:.5f} (x{radius_scale:.2f})")
    
    mesh_sampled = pu.particles_to_mesh(
        positions=particles_sampled,
        radius=radius_sampled,
        backend='splashsurf'
    )
    
    # 5. Save Results
    path_full = os.path.join(output_dir, "ref_full.obj")
    path_sampled = os.path.join(output_dir, "test_downsampled.obj")
    
    mesh_full.export(path_full)
    mesh_sampled.export(path_sampled)
    
    # Save Particles as NPY
    path_parts_full = os.path.join(output_dir, "ref_parts_full.npy")
    path_parts_sampled = os.path.join(output_dir, "test_parts_sampled.npy")
    np.save(path_parts_full, particles)
    np.save(path_parts_sampled, particles_sampled)
    
    print(f"\nMeshes saved to: {output_dir}")
    print(f"  Reference: {len(mesh_full.vertices)} verts")
    print(f"  Sampled:   {len(mesh_sampled.vertices)} verts")
    print("\nGeneration Complete. Now run: python agforge/visualize_downsampling.py --view")


def mode_view(args):
    print("=== MODE: VIEW (VTK) ===")
    
    # Check for meshes
    output_dir = "data/debug_meshes"
    path_full = os.path.join(output_dir, "ref_full.obj")
    path_sampled = os.path.join(output_dir, "test_downsampled.obj")
    path_parts_full = os.path.join(output_dir, "ref_parts_full.npy")
    path_parts_sampled = os.path.join(output_dir, "test_parts_sampled.npy")
    
    if not os.path.exists(path_full) or not os.path.exists(path_sampled):
        print(f"Error: Mesh files not found in {output_dir}")
        print("Please run with --generate first.")
        return

    try:
        import vtk
        from vtk.util import numpy_support
    except ImportError:
        print("Error: VTK not found. Please ensure 'vtk' is in your environment.")
        return

    def load_obj_actor(path, color):
        reader = vtk.vtkOBJReader()
        reader.SetFileName(path)
        reader.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        
        # Rotate 90 deg around X to fix orientation (standard OBJ issue)
        actor.RotateX(90)
        return actor

    def load_points_actor(path, color, radius=0.002):
        if not os.path.exists(path):
            print(f"Warning: Particle file not found: {path}")
            return None
            
        print(f"Loading particles from {path}...")
        try:
            points_np = np.load(path)
            # VTK requires contiguous memory and specific types
            points_np = np.ascontiguousarray(points_np, dtype=np.float32)
            print(f"  -> Loaded {points_np.shape[0]} particles. Range: {points_np.min(axis=0)} to {points_np.max(axis=0)}")
            
            points = vtk.vtkPoints()
            points.SetData(numpy_support.numpy_to_vtk(points_np, deep=True))
            
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            
            # CRITICAL FIX: Add Vertex cells so points are actually rendered
            # vtkPolyData with just points but no cells renders nothing.
            glyph = vtk.vtkVertexGlyphFilter()
            glyph.SetInputData(polydata)
            glyph.Update()
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            actor.GetProperty().SetPointSize(5)
            actor.GetProperty().RenderPointsAsSpheresOn() 
            
            # Rotate 90 deg around X to match mesh
            actor.RotateX(90)
            return actor
        except Exception as e:
            print(f"  -> Error loading particles: {e}")
            return None

    # Create Renderer 1 (Left - Full)
    renderer_L = vtk.vtkRenderer()
    renderer_L.SetViewport(0.0, 0.0, 0.5, 1.0) # Left half
    renderer_L.SetBackground(0.1, 0.1, 0.1)
    
    actor_full = load_obj_actor(path_full, (0.2, 0.2, 0.8)) # Blue
    actor_full.GetProperty().SetOpacity(0.5) # Translucent to see particles inside
    renderer_L.AddActor(actor_full)
    
    actor_p_full = load_points_actor(path_parts_full, (1.0, 0.5, 0.5)) # Reddish
    if actor_p_full: renderer_L.AddActor(actor_p_full)
    
    renderer_L.ResetCamera()
    
    # Create Renderer 2 (Right - Sampled)
    renderer_R = vtk.vtkRenderer()
    renderer_R.SetViewport(0.5, 0.0, 1.0, 1.0) # Right Half
    renderer_R.SetBackground(0.15, 0.15, 0.15) # Slightly lighter to distinguish
    
    actor_sampled = load_obj_actor(path_sampled, (0.2, 0.8, 0.2)) # Green
    actor_sampled.GetProperty().SetOpacity(0.5) # Translucent mesh to see particles
    renderer_R.AddActor(actor_sampled)
    
    actor_p_sampled = load_points_actor(path_parts_sampled, (1.0, 1.0, 0.0)) # Yellow
    if actor_p_sampled: renderer_R.AddActor(actor_p_sampled)
    
    # Sync Cameras (Optional but nice)
    renderer_R.SetActiveCamera(renderer_L.GetActiveCamera())

    # Render Window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1200, 600)
    render_window.AddRenderer(renderer_L)
    render_window.AddRenderer(renderer_R)
    render_window.SetWindowName("Comparison: Left (Original) vs Right (Downsampled)")

    # Interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Style (Trackball works best)
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    print("\nViewer Running (VTK).")
    print("LEFT  : Original (Blue)")
    print("RIGHT : Downsampled (Green)")
    print("\nControls:")
    print("  [m] Toggle Meshes")
    print("  [p] Toggle Particles")
    print("  Left-click: Rotate | Right-click: Zoom | Middle-click: Pan")

    def key_press_func(obj, event):
        key = obj.GetKeySym()
        if key == "m":
            if actor_full: actor_full.SetVisibility(not actor_full.GetVisibility())
            if actor_sampled: actor_sampled.SetVisibility(not actor_sampled.GetVisibility())
            print(f"Meshes Visible: {actor_full.GetVisibility()}")
        elif key == "p":
            if actor_p_full: actor_p_full.SetVisibility(not actor_p_full.GetVisibility())
            if actor_p_sampled: actor_p_sampled.SetVisibility(not actor_p_sampled.GetVisibility())
            print(f"Particles Visible: {actor_p_full.GetVisibility() if actor_p_full else 'N/A'}")
        
        render_window.Render()

    interactor.AddObserver("KeyPressEvent", key_press_func)
    
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Particle Downsampling Validation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true", help="Run simulation and generate meshes (Headless)")
    group.add_argument("--view", action="store_true", help="View generated meshes")
    
    args = parser.parse_args()
    
    if args.generate:
        mode_generate(args)
    elif args.view:
        mode_view(args)
