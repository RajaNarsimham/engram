import time

from engram.consolidation.engine import ConsolidationEngine


def _engine(**kw):
    # driver/registry/embed/gate are unused when on_process is stubbed
    return ConsolidationEngine(driver=None, registry=None, embed_fn=None, gate=None, **kw)


def _wait_for(pred, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_manual_starts_no_worker():
    e = _engine(trigger="manual")
    e.start()
    assert e._worker is None


def test_threshold_triggers_background_processing():
    calls = []
    e = _engine(trigger="threshold", threshold=2, on_process=lambda: (calls.append(1), [])[1])
    e.start()
    assert e._worker is not None
    e.enqueue("a")
    assert calls == []                      # below threshold, nothing yet
    e.enqueue("b")                          # reaches threshold -> wakes the worker
    assert _wait_for(lambda: calls)
    e.stop()
    assert calls == [1]


def test_scheduled_processes_on_interval():
    calls = []
    e = _engine(trigger="scheduled", interval=0.03, on_process=lambda: (calls.append(1), [])[1])
    e.start()
    e.enqueue("a")
    assert _wait_for(lambda: calls)
    e.stop()


def test_enqueue_below_threshold_does_not_wake():
    e = _engine(trigger="threshold", threshold=5)
    e.enqueue("a")
    assert not e._wake.is_set()
