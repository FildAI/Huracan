import hashlib
import json
import uuid


def hash256(data):
    canonical_data = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical_data.encode('utf-8')).hexdigest()


def hash_uuid(data):
    canonical_data = json.dumps(data, sort_keys=True, ensure_ascii=False)
    hash_obj = hashlib.sha256(canonical_data.encode('utf-8')).digest()
    return str(uuid.UUID(bytes=hash_obj[:16]))

