import os
import sys
import subprocess
import argparse
import platform
import shutil
from pathlib import Path

def get_nuitka_cmd(target_os, compiler="msvc", jobs=None, no_console=False):
    """Constructs the Nuitka command with necessary flags."""
    cmd = [
        sys.executable, "-m", "nuitka",
        "--mode=standalone",
        "--main=agforge/teleop_socket.py",
        
        # Output configuration
        f"--output-dir=dist/nuitka/{target_os}",
        "--output-filename=teleop_socket",
        
        "--show-progress",
        "--assume-yes-for-downloads",
        
        # Performance & Plugins
        "--lto=yes",
        "--enable-plugin=torch",
        "--enable-plugin=numpy",
        
        # Data & Core Packages
        "--include-data-dir=agforge/pbs_samples=agforge/pbs_samples",
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
        if no_console:
             cmd.append("--disable-console")
        
        cmd.append("--disable-ccache")
        
        if compiler == "mingw64":
            cmd.append("--mingw64")
        else:
            # MSVC Configuration for High-Mem Projects
            cmd.append("--msvc=latest")
            cmd.append("--low-memory")
            if jobs is None:
                jobs = 1 # Force serial build by default for MSVC to save memory
    
    return cmd

def get_pyinstaller_cmd(target_os, no_console=False):
    """Constructs the PyInstaller command."""
    sep = os.pathsep
    
    # Output Isolation
    dist_path = Path("dist/pyinstaller") / target_os
    work_path = Path("build/pyinstaller") / target_os
    
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
        f"--add-data={Path('agforge/pbs_samples').absolute()}{sep}pbs_samples",
        
        # Hidden Imports & Assets
        "--hidden-import=genesis",
        "--hidden-import=genesis.utils.mesh",
        "--hidden-import=agforge",
        "--hidden-import=websockets.legacy.server",
        "--hidden-import=tifffile",
        "--hidden-import=trimesh", 
        "--hidden-import=scipy.spatial.transform._rotation_groups",
        
        "--collect-all=coacd",
        "--collect-all=gstaichi",
        "--collect-all=z3",
        "--collect-all=glfw",
        "--collect-all=mujoco",
        "--collect-all=genesis", 
    ]
    
    if no_console:
        cmd.append("--windowed") # Equivalent to --noconsole, prevents cmd window on Windows
        
    return cmd

def run_local_build(tool, nuitka_compiler="msvc", jobs=None, no_console=False):
    """Run the build locally using the specified tool."""
    current_os = platform.system().lower()
    
    # Define output directory for cleanup
    # logic matches the commands above
    output_dir = Path("dist") / tool / current_os
    
    print(f"🚀 Starting Local Build using {tool.upper()} for {current_os}...")
    if tool == "nuitka" and current_os == "windows":
        print(f"🔧 Compiler Backend: {nuitka_compiler.upper()}")
    if jobs:
         print(f"⚙️  Parallel Jobs: {jobs}")
    print(f"🖥️  Console: {'Disabled' if no_console else 'Enabled'}")

    print(f"📂 Output Directory: {output_dir}")
    
    print(f"📂 Output Directory: {output_dir}")
    
    # Clean previous build
    dirs_to_clean = [output_dir]
    
    # Also clean intermediate build directory if applicable
    if tool == "pyinstaller":
        build_dir = Path("build") / tool / current_os
        dirs_to_clean.append(build_dir)

    for d in dirs_to_clean:
        if d.exists():
            print(f"🧹 Cleaning {d}...")
            try:
                # Retry logic for Windows file locks
                shutil.rmtree(d, ignore_errors=False)
            except PermissionError:
                print(f"⚠️  Warning: Could not fully clean {d}. Files might be in use.")
                print("   Attempting to continue...")

    if tool == "nuitka":
        cmd = get_nuitka_cmd(current_os, compiler=nuitka_compiler, jobs=jobs, no_console=no_console)
    elif tool == "pyinstaller":
        cmd = get_pyinstaller_cmd(current_os, no_console=no_console)
    else:
        print(f"❌ Unknown tool: {tool}")
        sys.exit(1)
    
    print(f"Executing: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print("✅ Build completed successfully.")
        
        # Helpful info on where to find it
        if tool == "pyinstaller":
            exe_path = output_dir / "teleop_socket" / ("teleop_socket.exe" if current_os == "windows" else "teleop_socket")
        else:
            # Nuitka structure varies slightly by mode, but usually:
            if current_os == "windows":
                exe_path = output_dir / "teleop_socket.dist" / "teleop_socket.exe"
            else:
                exe_path = output_dir / "teleop_socket.dist" / "teleop_socket.bin"
                
        print(f"👉 Executable located at: {exe_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed with error code {e.returncode}")
        sys.exit(1)

def run_docker_build():
    """Run the build using Docker (targets Linux output)."""
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
        output_dir = Path("dist/linux")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        subprocess.check_call([
            "docker", "cp",
            f"{container_id}:/app/teleop_socket",
            str(output_dir / "teleop_socket.bin")
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
    parser.add_argument("--no-console", action="store_true",
                        help="Disable terminal window (default: console enabled)")
    
    args = parser.parse_args()
    
    if args.target == "local":
        run_local_build(args.tool, nuitka_compiler=args.nuitka_compiler, jobs=args.jobs, no_console=args.no_console)
    elif args.target == "docker":
        if args.tool != "nuitka":
            print("⚠️  Warning: Docker build currently only supports Nuitka found in Dockerfile.")
        run_docker_build()
if __name__ == "__main__":
    main()
