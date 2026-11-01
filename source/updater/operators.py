import bpy
from bpy.types import Operator
import urllib.request
import json
import os
import zipfile
import shutil
import tempfile

GITHUB_REPO_URL = "https://github.com/DPN-dpn/XXMI-Custom-Physics"
GITHUB_API_LATEST_URL = "https://api.github.com/repos/DPN-dpn/XXMI-Custom-Physics/releases/latest"


def _redraw_ui_regions(context):
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                for region in area.regions:
                    if region.type == "UI":
                        region.tag_redraw()


class XXMI_PHYS_OT_check_update(Operator):
    bl_idname = "xxmi_phys.check_update"
    bl_label = "업데이트 체크"
    bl_description = "XXMI Custom Physics 최신 릴리스 버전이 있는지 확인합니다"

    def execute(self, context):
        try:
            with urllib.request.urlopen(GITHUB_API_LATEST_URL, timeout=5) as response:
                data = json.loads(response.read().decode())
                latest_version = data.get("tag_name", "").lstrip("v")
                context.scene["xxmi_phys_latest_version"] = latest_version

                from ... import bl_info

                current_version = ".".join(map(str, bl_info["version"]))
                context.scene["xxmi_phys_current_version"] = current_version
                context.scene["xxmi_phys_show_restart"] = False
                if latest_version and latest_version != current_version:
                    context.scene["xxmi_phys_update_available"] = True
                    self.report(
                        {"INFO"}, f"업데이트 가능: {current_version} → {latest_version}"
                    )
                else:
                    context.scene["xxmi_phys_update_available"] = False
                    self.report({"INFO"}, "최신 버전입니다.")
        except Exception as e:
            context.scene["xxmi_phys_update_available"] = False
            self.report({"ERROR"}, f"업데이트 확인 실패: {e}")

        _redraw_ui_regions(context)
        return {"FINISHED"}


class XXMI_PHYS_OT_do_update(Operator):
    bl_idname = "xxmi_phys.do_update"
    bl_label = "업데이트"
    bl_description = "애드온을 최신 버전으로 자동 다운로드 및 업데이트합니다"

    def execute(self, context):
        temp_dir = None
        try:
            with urllib.request.urlopen(GITHUB_API_LATEST_URL, timeout=5) as response:
                data = json.loads(response.read().decode())
                assets = data.get("assets", [])
                zip_url = ""
                zip_name = ""
                for asset in assets:
                    name = asset.get("name", "")
                    if name.endswith(".zip"):
                        zip_name = name
                        zip_url = asset.get("browser_download_url", "")
                        break
                if not zip_url:
                    self.report({"ERROR"}, "릴리스 zip 파일을 찾을 수 없습니다.")
                    return {"CANCELLED"}

            addon_name = os.path.splitext(zip_name)[0]

            # 애드온 루트 폴더 찾기
            current_dir = os.path.dirname(os.path.abspath(__file__))
            addons_root = None
            cur = current_dir
            while True:
                if os.path.basename(cur).lower() == "addons":
                    addons_root = cur
                    break
                parent = os.path.dirname(cur)
                if parent == cur:
                    break
                cur = parent

            if addons_root:
                addon_dir = os.path.join(addons_root, addon_name)
            else:
                # source/updater/operators.py 기준 상위 3단계가 애드온 루트
                addon_dir = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )

            temp_dir = tempfile.mkdtemp(prefix="xxmi_phys_update_")
            zip_path = os.path.join(temp_dir, zip_name or "update.zip")
            urllib.request.urlretrieve(zip_url, zip_path)

            extract_dir = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            entries = os.listdir(extract_dir)
            if not entries:
                raise Exception("압축을 풀었으나 내부에 항목이 없습니다.")

            new_addon_src = None
            if len(entries) == 1 and os.path.isdir(
                os.path.join(extract_dir, entries[0])
            ):
                first_level = os.path.join(extract_dir, entries[0])
                candidate = os.path.join(first_level, addon_name)
                if os.path.isdir(candidate):
                    new_addon_src = candidate
                elif os.path.basename(first_level) == addon_name:
                    new_addon_src = first_level

            if not new_addon_src:
                for root, dirs, _files in os.walk(extract_dir):
                    if addon_name in dirs:
                        new_addon_src = os.path.join(root, addon_name)
                        break

            if not new_addon_src:
                self.report(
                    {"ERROR"}, f"새 버전 애드온 폴더({addon_name})를 찾지 못했습니다."
                )
                shutil.rmtree(temp_dir, ignore_errors=True)
                return {"CANCELLED"}

            if os.path.exists(addon_dir):
                for name in os.listdir(addon_dir):
                    target = os.path.join(addon_dir, name)
                    try:
                        if os.path.isfile(target) or os.path.islink(target):
                            os.remove(target)
                        elif os.path.isdir(target):
                            shutil.rmtree(target)
                    except Exception as e:
                        self.report({"ERROR"}, f"{target} 삭제 실패: {e}")
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        return {"CANCELLED"}

            for name in os.listdir(new_addon_src):
                src = os.path.join(new_addon_src, name)
                dst = os.path.join(addon_dir, name)
                try:
                    shutil.move(src, dst)
                except Exception as e:
                    self.report({"ERROR"}, f"{src} 이동 실패: {e}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return {"CANCELLED"}

            if os.path.exists(zip_path):
                os.remove(zip_path)
            shutil.rmtree(temp_dir, ignore_errors=True)

            self.report({"WARNING"}, "블렌더를 재시작해 애드온을 새로고침해 주세요.")
            context.scene["xxmi_phys_show_restart"] = True
            _redraw_ui_regions(context)
            return {"FINISHED"}

        except Exception as e:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            self.report({"ERROR"}, f"업데이트 실패: {e}")
            context.scene["xxmi_phys_show_restart"] = False
            return {"CANCELLED"}


class XXMI_PHYS_OT_open_github(Operator):
    bl_idname = "xxmi_phys.open_github"
    bl_label = "GitHub"
    bl_description = "XXMI Custom Physics의 GitHub 저장소 페이지를 엽니다"

    def execute(self, context):
        import webbrowser

        webbrowser.open(GITHUB_REPO_URL)
        self.report({"INFO"}, "XXMI-Custom-Physics GitHub 페이지를 엽니다.")
        return {"FINISHED"}


classes = (
    XXMI_PHYS_OT_check_update,
    XXMI_PHYS_OT_do_update,
    XXMI_PHYS_OT_open_github,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

