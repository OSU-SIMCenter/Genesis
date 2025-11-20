import trimesh
import matplotlib.pyplot as plt

# Load mesh
mesh = trimesh.load('meshes/proc_cyl_stock.obj')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x, y, z = mesh.vertices.T
faces = mesh.faces

ax.plot_trisurf(x, y, z, triangles=faces, color='lightgray', edgecolor='k', alpha=0.7)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()  # Interactive window with mouse control for rotation/zoom/pan
