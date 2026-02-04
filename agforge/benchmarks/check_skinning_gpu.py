
import time
import torch
import genesis as gs
from agforge.options import TeleopOptions
from agforge.agforge_builder import build_env
from agforge.reconstruction import SurfaceReconstructor

def check_skinning_performance():
    gs.init(backend=gs.gpu)
    
    opts = TeleopOptions()
    opts.general.show_viewer = False
    print("Building environment...")
    env = build_env(opts)
    env.reset()
    
    # Configure Reconstructor directly
    reconstructor = SurfaceReconstructor(env)
    reconstructor.recon_enabled = True
    reconstructor.skinning_enabled = True
    reconstructor.recon_frame_interval = 1
    
    # Initialize skinning (requires one full reconstruction first)
    print("Initializing reconstruction...")
    reconstructor.create_reconstructed_mesh()
    reconstructor.init_skinning()
    
    # Run loop
    n_steps = 100
    print(f"Running {n_steps} skinning updates...")
    
    # Warmup
    for _ in range(10):
        reconstructor.update_skinning()
    torch.cuda.synchronize()
    
    start_time = time.time()
    for i in range(n_steps):
        # Fake frame increment
        reconstructor._global_frame += 1
        reconstructor.update_skinning()
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_steps * 1000
    print(f"Average Skinning Time: {avg_time:.4f} ms")

if __name__ == "__main__":
    check_skinning_performance()
