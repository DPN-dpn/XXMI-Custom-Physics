import json


def get_rigid_data(item):
    """RigidItem의 JSON 문자열 데이터를 딕셔너리로 파싱합니다."""
    if not item or not item.data_json:
        return {}
    try:
        return json.loads(item.data_json)
    except Exception:
        return {}


def set_rigid_data(item, data):
    """RigidItem의 JSON 문자열 데이터를 갱신하고 정점 수를 동기화합니다."""
    if not item:
        return
    item.data_json = json.dumps(data)
    item.vert_count = sum(len(v) for v in data.values())
