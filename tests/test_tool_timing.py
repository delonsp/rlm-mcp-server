from rlm_mcp import http_server as hs


def test_tool_timing_records_duration_and_error(monkeypatch):
    ticks = iter([10.0, 10.125])
    monkeypatch.setattr(hs.time, 'perf_counter', lambda: next(ticks))
    response = hs.call_tool('rlm_read', {'var_name': 'missing'})
    assert response.get('isError')
    stats = hs.metrics_collector.get_snapshot().tool_timing_by_name['rlm_read']
    assert stats['count'] == 1 and stats['avg_ms'] == 125


def test_dispatch_wait_is_measured_separately(monkeypatch):
    ticks = iter([10.0, 10.05])
    monkeypatch.setattr(hs.time, 'perf_counter', lambda: next(ticks))
    monkeypatch.setattr(hs, 'handle_mcp_request', lambda *a, **kw: None)
    hs.handle_mcp_request_locked(None)
    stats = hs.metrics_collector.get_snapshot().dispatch_wait_ms
    assert stats['count'] == 1 and stats['avg_ms'] == 50
