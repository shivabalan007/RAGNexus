from ragnexus.advanced_retrieval.models import Evidence
from ragnexus.advanced_retrieval.fusion import reciprocal_rank_fusion

def test_rrf_promotes_items_relevant_in_multiple_rankings():
    a = Evidence("a", "d1", "A")
    b = Evidence("b", "d1", "B")
    c = Evidence("c", "d2", "C")
    result = reciprocal_rank_fusion([[a, b], [b, c]])
    assert result[0].chunk_id == "b"
    assert [x.rank for x in result] == [1, 2, 3]
