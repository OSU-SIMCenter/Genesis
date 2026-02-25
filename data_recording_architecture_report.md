# Genesis Forging Simulation: Data Recording Architecture Report
*Generated for External Architecture Review (Feb 2026)*

## 1. Project Context & Objectives
This project is an advanced, high-fidelity robotic forging simulation built on top of the **Genesis** physics engine (genesis-world). Genesis was chosen because it uniquely enables the tight coupling of rigid body dynamics (the robot manipulator and tools) with the Material Point Method (MPM) solver (the deformable metal workpiece). 

**Primary Objective:** The end goal is to use this simulation to generate massive amounts of synthetic training data for Generalist Robot Policies (e.g., Vision-Language-Action models). These models must learn the complex, thermodynamic, and elastoplastic processes of metal forging.

## 2. The Data Infrastructure Challenge
The "Forging" use case presents unique data storage bottlenecks:
1. **High Volume & Velocity**: The simulation runs incredibly fast, producing high-frequency frames. Each frame contains state data for tens to hundreds of thousands of MPM particles (position, velocity, deformation gradients ($F$), plastic deformation scalars ($J_p$)).
2. **Variable Topology (Ragged Arrays)**: Operations like cutting, trimming, or adaptive particle resampling dynamically change the number of particles ($N$) mid-simulation. Standard ML tensors expect fixed dimensions.
3. **Training vs. Replay**: The data must be optimized for downstream ML training (high throughput read) rather than sequential debugging replay.

## 3. Current Implementation Stack
The current data architecture leverages heavily optimized, local-disk-first technologies. It was designed as a "Platinum Standard" for local/cluster generation speed.

| Component | Tool / Library | Role & Justification |
| :--- | :--- | :--- |
| **Physics Engine** | `genesis` | Unified, differentiable GPU physics. |
| **Data Format** | HDF5 (`h5py`) | Stores dense particle data. We use **One File Per Episode** (e.g., `ep_000000.h5`) to avoid HDF5's notorious file-locking contention. |
| **Compression** | LZF (in HDF5) | Lightning-fast compression that reduces array sizes significantly with almost zero CPU overhead during the simulation loop. |
| **Metadata Index** | `duckdb` | A single SQLite-like database (`metadata.duckdb`) that indexes all HDF5 files. Contains episode metadata (ID, path, success flag, duration, max_stress) allowing instant querying for DataLoaders without opening thousands of `.h5` files. |
| **Environment** | `pixi` | Used for reproducible package management (Python 3.10+, PyTorch, etc.). |

## 4. Key Implementation Details & Features
The core logic resides in `agforge/recorder.py` (`AgForgeRecorder` class) and is integrated into `agforge/teleop_socket.py`.

### A. Ragged Array Support (Padding + Active Mask)
To solve the variable particle count issue without crashing HDF5 or introducing complex Awkward Arrays:
- The recorder buffers frames in RAM during an episode.
- On flush, it calculates the **maximum particle count ($N_{max}$)** for that specific episode.
- All frames are padded to $N_{max}$ with zeros.
- An Active Mask dataset (`state/mpm_mask` of type Boolean) is saved alongside the data to map which particles in the padded array are physically real.
- *Efficiency:* The LZF compressor inherently squeezes the padded zero regions down to almost zero bytes on disk.

### B. RAM Buffering & "Undo" Logic
To support human-in-the-loop data generation (teleop):
- Data is purely buffered in RAM lists while the robot is pressing/striking.
- If the human user makes a mistake and hits "Undo" in the client, the `AgForgeRecorder` simply `pops()` the last strike's frames from the RAM buffer arrays. The mistake is never committed to disk.

### C. Directory Sharding
To prevent the OS "Inode Explosion" (where `ls` or Windows Explorer freezes when a directory has >10,000 files), the recorder automatically shards saved episodes into subdirectories: `data/train/chunk_000/ep_000000.h5`.

### D. HDF5 Schema Layout
Each episode file contains:
- `state/mpm_particles` - `(Time, N_{max}, 3)`
- `state/mpm_stress` - `(Time, N_{max})` (Using $J_p$ or Von Mises)
- `state/mpm_mask` - `(Time, N_{max})` (Boolean active mask)
- `obs/force_torque` - `(Time, 6)` (Robot EE or Joint forces)
- `obs/qpos` - `(Time, #_joints)` (Robot proprioception)
- `File Attributes` - `target_id`, `sim_version`, etc.

## 5. Alternative Architectures Considered (And Rejected)
Prior to finalizing this stack, we deeply evaluated a **Zarr v3 + Apache Parquet** architecture (the "Unified Forging Data Fabric"). 

**Why Zarr+Parquet was proposed:**
- It is the emerging standard for open-source robotics (Hugging Face LeRobot, Open X-Embodiment).
- Zarr is completely Cloud-Native (S3 compatible) and naturally handles extreme distributed parallel reads/writes because each chunk is a separate object.

**Why we rejected Zarr+Parquet (For Now):**
- **Local Performance vs Cloud:** This project prioritizes blazing fast *local simulation and training* on a workstation/local cluster. 
- **The Inode Problem:** Generating 100,000 episodes with Zarr's chunk-per-file nature would result in *millions* of tiny files scattered across the local disk, absolutely devastating local OS filesystem performance (Windows/Ext4).
- We concluded that HDF5 with padding + DuckDB is superior for our local simulation velocity. If cloud deployment happens later, an `h5 -> zarr` conversion script could easily bridge the gap.

## 6. Next Steps & Areas for Review
We are looking for an external model to critique this implementation in the context of late 2025/2026 MLOps and Robotics standards. 

**Specific questions for the reviewer:**
1. **The Padding/Masking Approach:** Is padding to $N_{max}$ and utilizing LZF compression still the most efficient way to handle ragged particle arrays in PyTorch/HDF5 pipelines, or has a newer PyTorch feature (like maturing `NestedTensors` or `Awkward Arrays`) made this obsolete for local training?
2. **DuckDB vs. Parquet:** Should we switch the `metadata.duckdb` index to a standard `index.parquet` for interoperability, even if we just use DuckDB to query the Parquet file?
3. **Zarr on Local Disks:** Have recent updates to Zarr v3 (e.g., sharding/ZipStores) solved the multi-million tiny file "inode explosion" issue on local filesystems? If so, should we reconsider Zarr to align with LeRobot conventions?
4. **Visualization:** We plan to integrate **Rerun.io** into the simulation loop to visually debug the MPM particle deformation (coloring point clouds by stress). Are there better/faster alternatives in 2026 for high-frequency physics debugging?
