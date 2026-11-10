from types import SimpleNamespace

from ragnexus.retrieval.fusion import reciprocal_rank_fusion


def chunk(identifier):
    return SimpleNamespace(id=identifier, content=f"chunk {identifier}")


def test_rrf_combines_rankings_and_deduplicates():
    a, b, c = chunk("a"), chunk("b"), chunk("c")
    result = reciprocal_rank_fusion(
        [(a, 0.9), (b, 0.8)],
        [(b, 12.0), (c, 9.0)],
    )
    assert [item.chunk.id for item in result] == ["b", "a", "c"]
    assert result[0].dense_score == 0.8
    assert result[0].bm25_score == 12.0
    assert result[0].rrf_score > result[1].rrf_score


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([], []) == []
