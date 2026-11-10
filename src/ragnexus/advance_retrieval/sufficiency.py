from .models import Evidence, RetrievalConfig

def assess_sufficiency(evidence: list[Evidence], config: RetrievalConfig) -> bool:
    if len(evidence) < config.min_evidence:
        return False
    distinct_docs = {str(item.document_id) for item in evidence}
    return len(distinct_docs) >= config.min_distinct_documents
