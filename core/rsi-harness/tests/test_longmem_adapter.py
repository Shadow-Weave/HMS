from rsi_harness import LongMemEvalAdapter, LongMemEvalItemError


def _item():
    return {
        "question_id": "q-small-1",
        "question": "What color did I prefer?",
        "question_date": "2025/01/03 (Fri) 10:00",
        "answer": "blue",
        "question_type": "single-session-user",
        "haystack_sessions": [
            [{"role": "user", "content": "I prefer blue.", "has_answer": True}],
            [{"role": "assistant", "content": "We discussed travel plans."}],
        ],
        "haystack_dates": ["2025/01/01 (Wed) 10:00", "2025/01/02 (Thu) 10:00"],
        "haystack_session_ids": ["s-1", "s-2"],
    }


def test_longmem_adapter_preserves_session_order_and_drops_answer_annotation():
    adapter = LongMemEvalAdapter()
    sessions = adapter.sessions(_item())
    query = adapter.query(_item(), evaluation_id="eval-1")

    assert [session.session_id for session in sessions] == ["s-1", "s-2"]
    assert [session.source_position for session in sessions] == [0, 1]
    assert "has_answer" not in sessions[0].messages[0]
    assert query.query_id == "q-small-1"
    assert query.query_position == 2
    assert not hasattr(query, "answer")


def test_longmem_adapter_can_bound_sessions_without_changing_query_position_contract():
    sessions = LongMemEvalAdapter().sessions(_item(), max_sessions=1)
    assert len(sessions) == 1
    assert sessions[0].source_position == 0


def test_longmem_adapter_rejects_misaligned_arrays():
    item = _item()
    item["haystack_dates"] = []
    try:
        LongMemEvalAdapter().sessions(item)
    except LongMemEvalItemError as exc:
        assert "aligned" in str(exc)
    else:
        raise AssertionError("misaligned LongMemEval item was accepted")


def test_longmem_runtime_bundle_excludes_reference_fields():
    runtime = LongMemEvalAdapter().runtime(_item(), evaluation_id="eval-1")
    assert runtime.item_id == "q-small-1"
    assert runtime.query.query_id == "q-small-1"
    assert not hasattr(runtime.query, "answer")
    assert all("has_answer" not in message for session in runtime.sessions for message in session.messages)
