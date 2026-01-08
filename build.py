import os
import sys
import subprocess
import argparse
import platform
import shutil
from pathlib import Path

def get_nuitka_cmd(target_os, compiler="msvc", jobs=None):
    """
    Constructs the Nuitka command with necessary flags for the specific target.
    """
    cmd = [
        sys.executable, "-m", "nuitka",
        "--mode=standalone",
        "--main=agforge/teleop_socket.py",
        
        # Output configuration (Manual Separation)
        f"--output-dir=dist/nuitka/{target_os}",
        "--output-filename=teleop_socket",
        
        "--show-progress",
        "--assume-yes-for-downloads",
        
        # Performance Optimizations
        "--lto=yes", # Link Time Optimization
        
        # Plugins
        "--enable-plugin=torch",
        "--enable-plugin=numpy",
        
        # Data Directories
        "--include-data-dir=agforge/pbs_samples=agforge/pbs_samples",
        
        # Core Packages
        "--include-package=genesis",
        "--include-package=agforge",
    ]

    if jobs:
        cmd.append(f"--jobs={jobs}")

    if target_os == "linux":
        # Linux specific flags for maximum compatibility
        cmd.extend([
            "--static-libpython=no", # Keep 'no' for glibc compatibility issues often seen with 'yes'
            "--include-module=websockets.legacy.server",
        ])
    elif target_os == "windows":
        cmd.extend([
            "--disable-console",
            "--disable-ccache",
        ])
        
        if compiler == "mingw64":
            cmd.append("--mingw64")
        else:
            # MSVC Configuration for High-Mem Projects (Torch)
            cmd.append("--msvc=latest")
            cmd.append("--low-memory") # Fixes C1002 heap error without hurting runtime perf
            if jobs is None:
                jobs = 1 # Force serial build by default for MSVC to save memory
    
    return cmd

def get_pyinstaller_cmd(target_os):
    """
    Constructs the PyInstaller command.
    """
    sep = ";" if target_os == "windows" else ":"
    
    # Output Isolation
    dist_path = os.path.join("dist", "pyinstaller", target_os)
    work_path = os.path.join("build", "pyinstaller", target_os)
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "agforge/teleop_socket.py",
        "--name=teleop_socket",
        "--clean",
        "--noconfirm",
        "--onedir", 
        
        # Output Paths
        f"--distpath={dist_path}",
        f"--workpath={work_path}",
        f"--specpath={work_path}", # Put spec file in build dir to keep root clean
        
        # Data Directories
        # Use absolute path to avoid ambiguity. Map to root 'pbs_samples' to match environment.py's resource_path logic.
        f"--add-data={os.path.abspath('agforge/pbs_samples')}{sep}pbs_samples",
        
        # Hidden Imports (Manual)
        # Note: --collect-all copies the package source/libs, but hidden-import ensures analysis finds it.
        "--hidden-import=genesis",
        "--hidden-import=genesis.utils.mesh", # Fix for crash in some environments
        "--hidden-import=agforge",
        "--hidden-import=websockets.legacy.server",
        "--hidden-import=tifffile",
        "--hidden-import=trimesh", 
        "--hidden-import=scipy.spatial.transform._rotation_groups",
        
        # Collect complex packages that rely on internal data/binaries
        "--collect-all=coacd",
        "--collect-all=gstaichi", # ESSENTIAL: Collects runtime_*.bc bitcode files for GPU/CPU backends
        "--collect-all=z3", # ESSENTIAL: Collects libz3.dll
        "--collect-all=glfw", # ESSENTIAL: Collects glfw3.dll
        "--collect-all=mujoco", # ESSENTIAL: Collects mujoco.dll and assets
        # genesis is collected to ensure all assets are present, despite being hidden-imported above
        "--collect-all=genesis", 
        
        "--windowed", # Equivalent to --noconsole, prevents cmd window on Windows
    ]
    
    return cmd

def run_local_build(tool, nuitka_compiler="msvc", jobs=None):
    """Run the build locally using the specified tool."""
    current_os = platform.system().lower()
    
    # Define output directory for cleanup
    # logic matches the commands above
    output_dir = os.path.join("dist", tool, current_os)
    
    print(f"🚀 Starting Local Build using {tool.upper()} for {current_os}...")
    if tool == "nuitka" and current_os == "windows":
        print(f"🔧 Compiler Backend: {nuitka_compiler.upper()}")
    if jobs:
         print(f"⚙️  Parallel Jobs: {jobs}")

    print(f"📂 Output Directory: {output_dir}")
    
    # 1. Clean previous build (Granular)
    # 1. Clean previous build (Granular)
    # Clean output directory (dist/)
    dirs_to_clean = [output_dir]
    
    # Also clean intermediate build directory if applicable
    if tool == "pyinstaller":
        # PyInstaller uses a separate workpath
        build_dir = os.path.join("build", tool, current_os)
        dirs_to_clean.append(build_dir)
    # Note: Nuitka with --output-dir puts its .build folder INSIDE that dir, 
    # so cleaning output_dir works for both.

    for d in dirs_to_clean:
        if os.path.exists(d):
            print(f"🧹 Cleaning {d}...")
            try:
                # Retry logic for Windows file locks
                shutil.rmtree(d, ignore_errors=False)
            except PermissionError:
                print(f"⚠️  Warning: Could not fully clean {d}. Files might be in use.")
                print("   Attempting to continue...")

    if tool == "nuitka":
        cmd = get_nuitka_cmd(current_os, compiler=nuitka_compiler, jobs=jobs)
    elif tool == "pyinstaller":
        cmd = get_pyinstaller_cmd(current_os)
    else:
        print(f"❌ Unknown tool: {tool}")
        sys.exit(1)
    
    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print("✅ Build completed successfully.")
        
        # Helpful info on where to find it
        if tool == "pyinstaller":
            exe_path = os.path.join(output_dir, "teleop_socket", "teleop_socket.exe" if current_os == "windows" else "teleop_socket")
        else:
            # Nuitka structure varies slightly by mode, but usually:
            if current_os == "windows":
                exe_path = os.path.join(output_dir, "teleop_socket.dist", "teleop_socket.exe")
            else:
                exe_path = os.path.join(output_dir, "teleop_socket.dist", "teleop_socket.bin")
                
        print(f"👉 Executable located at: {exe_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed with error code {e.returncode}")
        sys.exit(1)

# ... (docker build skipped for brevity, assumed unchanged) ...

def run_docker_build():
    """Run the build using Docker (targets Linux output)."""
    # ... (content remains same)
    print("🐳 Starting Docker Build (Linux Target)...")
    
    # Check if docker is available
    if shutil.which("docker") is None:
        print("❌ Docker executable not found. Please install Docker.")
        sys.exit(1)

    cmd = [
        "docker", "build",
        "-t", "teleop_builder",
        "-f", "Dockerfile",
        "."
    ]
    
    print("Building Docker image...")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print("❌ Docker build failed.")
        sys.exit(1)
        
    print("Extracting artifacts from container...")
    container_id = subprocess.check_output(
        ["docker", "create", "teleop_builder"]
    ).decode().strip()
    
    try:
        if os.path.exists("dist/linux"):
            shutil.rmtree("dist/linux")
        os.makedirs("dist/linux", exist_ok=True)
        
        subprocess.check_call([
            "docker", "cp",
            f"{container_id}:/app/teleop_socket",
            "dist/linux/teleop_socket.bin"
        ])
        print("✅ Artifact extracted to dist/linux/teleop_socket.bin")
    finally:
        subprocess.call(["docker", "rm", "-v", container_id])

def main():
    parser = argparse.ArgumentParser(description="Build automation for Teleop Socket")
    parser.add_argument("--target", choices=["local", "docker"], default="local",
                        help="Build target: 'local' (native OS) or 'docker' (Linux container)")
    parser.add_argument("--tool", choices=["nuitka", "pyinstaller"], default="pyinstaller",
                        help="Build tool to use for local builds (default: nuitka)")
    parser.add_argument("--nuitka-compiler", choices=["msvc", "mingw64"], default="msvc",
                        help="Compiler for Nuitka on Windows: 'msvc' (default, system VS) or 'mingw64' (downloaded)")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Number of parallel jobs for Nuitka (reduces memory usage)")
    
    args = parser.parse_args()
    
    if args.target == "local":
        run_local_build(args.tool, nuitka_compiler=args.nuitka_compiler, jobs=args.jobs)
    elif args.target == "docker":
        if args.tool != "nuitka":
            print("⚠️  Warning: Docker build currently only supports Nuitka found in Dockerfile.")
        run_docker_build()
if __name__ == "__main__":
    main()
