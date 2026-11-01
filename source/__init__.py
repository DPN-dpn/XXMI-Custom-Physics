from . import utils
from . import properties
from . import breast
from . import simulation
from . import injector
from . import updater

modules = [
    utils,
    properties,
    breast,
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
