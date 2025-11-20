import numpy as np
import genesis as gs
import json
import time

class GenesisEnvironment():
    def __init__(self):
        ########################## init ##########################
        gs.init(backend=gs.gpu)

        ########################## create a scene ##########################
        self.scene = gs.Scene(
            viewer_options = gs.options.ViewerOptions(
                res           = (1920, 1200),
                camera_pos    = (0.5, -3.5, 2.5),
                camera_lookat = (0.5, 0.0, 1.0),
                camera_fov    = 60,
                max_FPS       = 60,
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound   = (-5.0, -2.0, -2.0),
                upper_bound   = (5.0, 2.0, 2.0),
                particle_size = 0.1,
            ),
            vis_options=gs.options.VisOptions(
                visualize_mpm_boundary = True,
            ),

            sim_options = gs.options.SimOptions(
                dt = 0.01,
                gravity=[0.0, 0.0, 0.0],
            ),
            show_viewer = True,
        )

        ########################## entities ##########################

        self.billet = self.scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(),
            morph=gs.morphs.Mesh(
                file="meshes/proc_cyl_stock.obj",
                scale=0.25,
                pos=(0.0, 0.0, 0.0),
            ),
            surface=gs.surfaces.Default(
                color    = (0.3, 0.3, 1.0),
                vis_mode = 'particle',
            ),
        )

        self.cam = self.scene.add_camera(
            res           = (1080, 720),
            pos           = (-0.5, -2, 1.5),
            lookat        = (-0.5, 0.0, 0.0),
            fov           = 60,
            GUI    = False,
        )

        ########################## build ##########################
        self.scene.build()


    def do_press(self):
        # currently identical to the update function, will change once presses are implemented
        vertices = self.billet.get_particles()[0]
        print("Got genesis result!")
        return {
            "Vertices": vertices.flatten().tolist(),
            "Steps": [0],
            "Temperatures": np.full(len(vertices), 293.0, dtype=float).tolist(),
            "Pressure": 0, # my stand in pressure value goes from 0 to 100
                            # feel free to change this, but message Jonah if you do
            "StressField": -1,
        }
    
    def temperature_result(self, count):
        return{
            "Temperatures": np.full(count, 293.0, dtype=float).tolist(),
            "Times": [10]
        }
    
    def update(self):
        self.scene.step()
        self.scene.step()
        vertices = self.billet.get_particles()[0]
        print("Got genesis result!")
        return {
            "Vertices": vertices.flatten().tolist(),
            "Steps": [0],
            "Temperatures": np.full(len(vertices), 293.0, dtype=float).tolist(),
            "Pressure": 0, # my stand in pressure value goes from 0 to 100
                            # feel free to change this, but message Jonah if you do
            "StressField": -1,
        }
