from remote_gym import serve

# Import the new wrapper and the config
from environment_gym import AgilityForgeGymEnv
from options import TrainingOptions

def make_env():
    """
    A 'maker' function that remote-gym can call to create
    a new environment instance for a client.
    """
    print("Creating a new environment instance...")
    cfg = TrainingOptions()
    
    # Override any settings for the server here
    cfg.general.record = False
    cfg.general.show_viewer = False
    
    return AgilityForgeGymEnv(cfg=cfg)

if __name__ == "__main__":
    print("Starting remote-gym server on port 5555...")
    print("A client can now connect and control the simulation.")
    
    # This single line creates and runs the server.
    # It automatically serves the 'step', 'reset', etc. methods
    # of the environment created by 'make_env'.
    serve(make_env, port=5555)