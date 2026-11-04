from . import utils
from . import properties
from . import bone
from . import rigid
from . import simulation
from . import injector
from . import updater

modules = [
    utils,
    properties,
    bone,
    rigid,
    simulation,
    injector,
    updater,
]


def register():
    for mod in modules:
        if hasattr(mod, "register"):
            mod.register()


def unregister():
    for mod in reversed(modules):
        if hasattr(mod, "unregister"):
            mod.unregister()
