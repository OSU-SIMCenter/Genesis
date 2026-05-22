import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_all_data(shard_path, episode_idx=0, output_img="all_data_plot.png"):
    if not os.path.exists(shard_path):
        print(f"File not found: {shard_path}")
        return

    with h5py.File(shard_path, 'r') as f:
        ep_keys = list(f.keys())
        if not ep_keys:
            print("No episodes found in shard.")
            return
            
        # We only want to plot the most recent episode
        ep_name = sorted(ep_keys)[-1]
        print(f"Visualizing only the most recent episode: {ep_name}")
        
        ep_grp = f[ep_name]
        obs_grp = ep_grp["observations"]["state"]["scene"]
        
        forces = obs_grp["force_torque"][:]
        qpos = obs_grp["qpos"][:]
        dof_cmds = ep_grp["actions"]["dof_velocity_cmd"][:]
        
        part_grp = ep_grp["observations"]["state"]["particles"]
        p_temp = part_grp["temp_values"][:]
        p_vel = part_grp["vel_values"][:]
        p_pos = part_grp["pos_values"][:]
        if "detF_values" in part_grp:
            p_detF = part_grp["detF_values"][:]
        elif "Jp_values" in part_grp:
            p_detF = part_grp["Jp_values"][:]
        else:
            p_detF = np.ones(p_temp.shape)
            
        p_vol = part_grp.attrs.get("particle_vol", 1.0) # default to 1.0 if not recorded
        offsets = part_grp["offsets"][:]
        
        num_steps = forces.shape[0]
        
        avg_temp = np.zeros(num_steps)
        max_temp = np.zeros(num_steps)
        min_temp = np.zeros(num_steps)
        avg_speed = np.zeros(num_steps)
        max_speed = np.zeros(num_steps)
        billet_width_y = np.zeros(num_steps)
        total_volume = np.zeros(num_steps)
        
        for i in range(num_steps):
            start_idx = offsets[i]
            end_idx = offsets[i+1]
            
            if start_idx < end_idx:
                frame_temp = p_temp[start_idx:end_idx]
                avg_temp[i] = np.mean(frame_temp)
                max_temp[i] = np.max(frame_temp)
                min_temp[i] = np.min(frame_temp)
                
                frame_vel = p_vel[start_idx:end_idx]
                speed = np.linalg.norm(frame_vel, axis=1)
                avg_speed[i] = np.mean(speed)
                max_speed[i] = np.max(speed)
                
                frame_pos = p_pos[start_idx:end_idx]
                billet_width_y[i] = np.max(frame_pos[:, 1]) - np.min(frame_pos[:, 1])
                
                frame_detF = p_detF[start_idx:end_idx]
                # True Volume = Sum( V_p0 * det(F) )
                total_volume[i] = np.sum(frame_detF) * p_vol
                
        time_steps = np.arange(num_steps)

        
        # Create a large figure with 7 subplots
        fig, axs = plt.subplots(7, 1, figsize=(12, 24), sharex=True)
        
        # 1. Forces
        axs[0].plot(time_steps, forces[:, 0], label="Left Gripper Force", color='r')
        axs[0].plot(time_steps, forces[:, 1], label="Right Gripper Force", color='b')
        axs[0].set_ylabel("Force")
        axs[0].set_title("Resistance Forces (Outliers Removed)")
        axs[0].grid(True, alpha=0.5)
        axs[0].legend()
        
        # Robust Y-limit for Forces to cut off extreme outliers
        f_flat = forces.flatten()
        f_valid = f_flat[np.abs(f_flat) > 1e-3]
        if len(f_valid) > 0:
            # Use 5th and 95th percentile of non-zero forces
            v_min, v_max = np.percentile(f_valid, 5), np.percentile(f_valid, 95)
            margin = max((v_max - v_min) * 0.2, 10.0)
            axs[0].set_ylim(v_min - margin, v_max + margin)
        
        # 2. Joint Positions
        axs[1].plot(time_steps, qpos[:, 0], label="Slider X", color='purple')
        axs[1].plot(time_steps, qpos[:, 2], label="Jaw L", color='r', linestyle='--')
        axs[1].plot(time_steps, qpos[:, 3], label="Jaw R", color='b', linestyle='--')
        axs[1].set_ylabel("Position")
        axs[1].set_title("Robot Joint Positions")
        axs[1].grid(True, alpha=0.5)
        axs[1].legend()
        
        # 3. Velocity Commands
        axs[2].plot(time_steps, dof_cmds[:, 0], label="Cmd Slider", color='purple')
        axs[2].plot(time_steps, dof_cmds[:, 2], label="Cmd Jaw L", color='orange')
        axs[2].plot(time_steps, dof_cmds[:, 3], label="Cmd Jaw R", color='green')
        axs[2].set_ylabel("Velocity Cmd")
        axs[2].set_title("Motor Velocity Commands")
        axs[2].grid(True, alpha=0.5)
        axs[2].legend()
        
        # 4. Particle Temperatures
        axs[3].plot(time_steps, max_temp, label="Max Temp", color='red')
        axs[3].plot(time_steps, avg_temp, label="Avg Temp", color='orange')
        axs[3].plot(time_steps, min_temp, label="Min Temp", color='blue')
        axs[3].set_ylabel("Temperature (C)")
        axs[3].set_title("Billet Temperature Distribution")
        axs[3].grid(True, alpha=0.5)
        axs[3].legend()
        
        # 5. Particle Speeds
        axs[4].plot(time_steps, max_speed, label="Max Particle Speed", color='magenta')
        axs[4].plot(time_steps, avg_speed, label="Avg Particle Speed", color='teal')
        axs[4].set_ylabel("Speed (m/s)")
        axs[4].set_title("Billet Deformation Speed")
        axs[4].grid(True, alpha=0.5)
        axs[4].legend()
        
        # 6. Billet Shape (Width / COM)
        axs[5].plot(time_steps, billet_width_y, label="Billet Width (Y axis)", color='brown')
        axs[5].set_ylabel("Width (m)")
        axs[5].set_title("Billet Deformation / Shape")
        axs[5].grid(True, alpha=0.5)
        axs[5].legend()
        
        # 7. Volume Conservation (Total Volume)
        if len(total_volume) > 0 and total_volume[0] > 0:
            volume_percent = (total_volume / total_volume[0]) * 100.0
        else:
            volume_percent = total_volume * 0.0 + 100.0 # fallback
            
        volume_lost_percent = 100.0 - volume_percent
            
        axs[6].plot(time_steps, volume_lost_percent, label="Volume Lost (%)", color='red')
        axs[6].set_ylabel("Volume Lost (%)")
        axs[6].set_xlabel("Time Step (Frames)")
        axs[6].set_title("Volume Conservation Error (%)")
        axs[6].grid(True, alpha=0.5)
        axs[6].legend()
        # Set y-limits around 0 to show the tiny error
        axs[6].set_ylim(-1.0, 1.0)
        
        plt.tight_layout()
        plt.savefig(output_img, dpi=150)
        print(f"Plot saved to {output_img}")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "data", "train", "shard_0000.h5")
    visualize_all_data(data_path)
