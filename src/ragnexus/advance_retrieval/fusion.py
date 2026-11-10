from collections import defaultdict
from .models import Evidence

def reciprocal_rank_fusion(rankings: list[list[Evidence]], rrf_k: int = 60) -> list[Evidence]:
    scores = defaultdict(float)
    objects = {}
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, start=1):
            key = (str(item.document_id), str(item.chunk_id))
            if key in seen:
                continue
            seen.add(key)
            scores[key] += 1.0 / (rrf_k + rank)
            objects[key] = item
    ordered = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [
        Evidence(
            chunk_id=objects[k].chunk_id,
            document_id=objects[k].document_id,
            content=objects[k].content,
            rank=i,
            score=scores[k],
            metadata=objects[k].metadata,
            parent_content=objects[k].parent_content,
        )
        for i, k in enumerate(ordered, start=1)
    ]
