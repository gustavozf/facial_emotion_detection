import json

def dump_json(fpath, data):
    with open(fpath, 'w', encoding='utf8') as _f:
        json.dump(data, _f, ensure_ascii=False, indent=2)