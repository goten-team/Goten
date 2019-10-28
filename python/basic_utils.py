import hashlib


def str_hash(s):
    return int(int(hashlib.sha224(s.encode('utf-8')).hexdigest(), 16) % ((1 << 62) - 1))
