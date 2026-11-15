from ragnexus.core.permissions import has_permission


def test_viewer_cannot_delete_documents():
    assert not has_permission("viewer", "documents:delete_any")
    assert not has_permission("viewer", "documents:delete_own")


def test_editor_can_delete_only_own_documents_by_policy():
    # The permission is intentionally delete_own; endpoint also compares uploaded_by to actor.
    assert has_permission("editor", "documents:delete_own")
    assert not has_permission("editor", "documents:delete_any")


def test_workspace_admin_can_delete_any_document():
    assert has_permission("admin", "documents:delete_any")
    assert has_permission("owner", "documents:delete_any")
