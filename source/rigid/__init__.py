from . import ui_list
from . import operators
from . import panel

modules = [
    ui_list,
    operators,
    panel,
]


def register():
    for mod in modules:
        if hasattr(mod, "register"):
            mod.register()


def unregister():
    for mod in reversed(modules):
        if hasattr(mod, "unregister"):
            mod.unregister()
