# calculate_pressing_speed.py

# --- Old Configuration ---
old_dt = 1.4e-6 * 8  # 1.12e-5 seconds per step
old_press_velocity = 30.0  # meters per second

# The distance the robot physically moved in a single integration step:
old_distance_per_step = old_press_velocity * old_dt

# --- New Configuration ---
new_dt = 1.0  # 1.0 seconds per step

# --- Calculations ---
# To maintain the exact same physical displacement per simulation step,
# the new velocity must cover the 'old_distance_per_step' over the span of 'new_dt'.
# Distance = Velocity * Time -> Velocity = Distance / Time
new_press_velocity = old_distance_per_step / new_dt

# Equivalently: new_press_velocity = old_press_velocity * (old_dt / new_dt)
velocity_scaling_factor = old_dt / new_dt

# --- Output ---
print("=== TIMESTEP SCALING ===")
print(f"Old dt: {old_dt} s")
print(f"New dt: {new_dt} s")
print(f"dt increased by: {(new_dt/old_dt):,.12f}x\n")

print("=== DISPLACEMENT ===")
print(f"Physical distance moved per step: {old_distance_per_step:,.12f} meters ({old_distance_per_step * 1000:.12f} mm)\n")

print("=== VELOCITY ADJUSTMENT ===")
print(f"Old Press Velocity: {old_press_velocity} m/s")
print(f"Required Velocity Scaling Factor: {velocity_scaling_factor:,.12f}x")
print(f"Required time scaling factor: {(1/velocity_scaling_factor):,.12f}x")
print(f"New Press Velocity: {new_press_velocity:,.12f} m/s")

print(30.0 * (1.4e-6 * 8))