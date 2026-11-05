from src.order_service import normalize_order_ids


def test_normalize_order_ids_removes_duplicates() -> None:
    assert set(normalize_order_ids([" a1 ", "A1", "b2"])) == {"A1", "B2"}


def test_normalize_order_ids_preserves_order() -> None:
    assert normalize_order_ids(["b2", "a1", "b2"]) == ["B2", "A1"]
