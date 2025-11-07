from flask import Flask, request, jsonify
import numpy as np
import warnings

# Import the new wrapper and the config
from environment_gym import AgilityForgeGymEnv
from options import TrainingOptions

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Create a SINGLE, GLOBAL environment instance ---
# This is crucial. The env holds the simulation state.
print("Building the simulation environment... (This may take a moment)")
cfg = TrainingOptions()
cfg.general.record = False
cfg.general.show_viewer = False
env = AgilityForgeGymEnv(cfg=cfg)
print("Environment ready. Server listening on port 5000.")


# --- 3. Define the API Endpoints ---
@app.route('/reset', methods=['POST'])
def reset():
    """Resets the environment."""
    obs, info = env.reset()
    return jsonify({
        "observation": obs.tolist(),
        "info": info
    })

@app.route('/step', methods=['POST'])
def step():
    """Steps the environment."""
    data = request.json
    action = np.array(data['action'])
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    return jsonify({
        "observation": obs.tolist(),
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info
    })

@app.route('/action_space', methods=['GET'])
def action_space():
    """Endpoint for a client to discover the action space."""
    space = env.action_space
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # Suppress inf warning
        return jsonify({
            "type": "Box",
            "low": space.low.tolist(),
            "high": space.high.tolist(),
            "shape": space.shape,
            "dtype": str(space.dtype)
        })

@app.route('/observation_space', methods=['GET'])
def observation_space():
    """Endpoint for a client to discover the observation space."""
    space = env.observation_space
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # Suppress inf warning
        # Replace inf with a large number for JSON compatibility
        low = np.where(np.isinf(space.low), -1e10, space.low).tolist()
        high = np.where(np.isinf(space.high), 1e10, space.high).tolist()
        return jsonify({
            "type": "Box",
            "low": low,
            "high": high,
            "shape": space.shape,
            "dtype": str(space.dtype)
        })

if __name__ == "__main__":
    # Run the Flask app
    # host='0.0.0.0' makes it accessible on your network
    # debug=False is important for performance
    app.run(host='0.0.0.0', port=5000, debug=False)