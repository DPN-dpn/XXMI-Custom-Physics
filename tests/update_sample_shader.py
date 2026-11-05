import os, sys
sys.path.insert(0, os.getcwd())
import bpy
from source.injector.hsr import export_vertex_physics_hlsl_hsr
from source.injector.buffer_io import _export_physics_state_buf

class DummyProps:
    strength = 1.0
    speed = 1.1
    damping = 0.85
    inertia = 1.2
    idle_motion = 0.4
    falloff_power = 1.0
    bone_items = []

props = DummyProps()
custom_physics_dir = "samples/Archer - 복사본/CustomPhysics"
vertex_count = 74422

# Bone 0 (Right): 24130, Bone 1 (Left): 23379
# In HSR shader: L = 23379, R = 24130
export_vertex_physics_hlsl_hsr(custom_physics_dir, 0, vertex_count, props, 1.0, root_vert_idx_l=23379, root_vert_idx_r=24130)
_export_physics_state_buf(custom_physics_dir, 0)

print("Updated CustomPhysics_0.hlsl and CustomPhysicsState_0.buf with independent left/right anchors!")
