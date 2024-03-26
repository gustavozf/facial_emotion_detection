import json

def dump_json(fpath, data):
    with open(fpath, 'w', encoding='utf8') as _f:
        json.dump(data, _f, ensure_ascii=False, indent=2)

def read_json(fpath):
    with open(fpath, 'r', encoding='utf8') as _f:
        data = json.load(_f)
    return data