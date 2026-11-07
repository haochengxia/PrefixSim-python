import json

import pytest

# LRU, 4 blocks, by hand (deepest-first replay, so a request's root is its newest block):
#   r0 1 2 3  miss all                LRU old->new: 3 2 1
#   r1 1 2 4  hit 1 2                 3 4 2 1
#   r2 1 2 3  hit all                 4 3 2 1
#   r3 5 6    miss; evict 4, 3        2 1 6 5
#   r4 1 2 4  hit 1 2; 1, 2 pinned, so evict 6
# blocks 14, hits 7, evictions 3.
LRU_TRACE = [[1, 2, 3], [1, 2, 4], [1, 2, 3], [5, 6], [1, 2, 4]]

# FIFO, 3 blocks: r2, r3 evict 2 then 1, so r4 misses 1 and hits 7, a gap.
# blocks 8; hits 2 (r1's 1, r4's 7); engine hits 1 (r1's 1).
GAP_TRACE = [[1, 2], [1, 7], [8], [9], [1, 7]]


def _jsonl(path, requests):
    path.write_text("".join(json.dumps({"timestamp": float(i), "hash_ids": b}) + "\n" for i, b in enumerate(requests)))
    return path


@pytest.fixture
def lru_trace(tmp_path):
    return _jsonl(tmp_path / "lru.jsonl", LRU_TRACE)


@pytest.fixture
def gap_trace(tmp_path):
    return _jsonl(tmp_path / "gap.jsonl", GAP_TRACE)
