bl_info = {
    "name": "XXMI Custom Physics",
    "author": "DPN",
    "version": (1, 2, 2),
    "blender": (2, 80, 0),
    "location": "N-Panel > XXMI Physics",
    "description": "XXMI 모드의 추가 파츠(가슴, 장발/포니테일/트윈테일, 스커트, 꼬리, 날개 등)에 가상 스켈레톤 기반 커스텀 물리 쉐이더를 주입하는 애드온",
    "category": "3D View",
}

from . import source


def register():
    source.register()


def unregister():
    source.unregister()


if __name__ == "__main__":
    register()
