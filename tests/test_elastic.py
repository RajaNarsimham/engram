import pytest

from engram.drivers.base import ArchInfo, BaseLLMDriver, DriverCapabilities
from engram.registry.registry import Capability, CapabilityKind


class MockDriver(BaseLLMDriver):
    adapter_dir = "."

    def capabilities(self):
        return DriverCapabilities(train_lora=False, tool_use=False)

    def arch_info(self):
        return ArchInfo(induction_capable=True)

    def generate(self, req):
        yield "x"

    def embed(self, texts):
        return [[0.1] for _ in texts]


def test_vllm_driver_module_imports_without_vllm():
    # the heavy vllm import is lazy (in __init__), so the module imports fine here
    from engram.drivers.vllm_driver import VLLMDriver
    assert VLLMDriver is not None


def test_multinode_shares_state_via_store(tmp_path):
    """Horizontal scale: node A's skills/tenants are visible to node B through the
    shared Store — nodes are stateless."""
    pytest.importorskip("moto")
    pytest.importorskip("boto3")
    import boto3
    from moto import mock_aws

    from engram.core import Engram
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="engram-elastic")
        cfg = {"type": "aws", "bucket": "engram-elastic", "region": "us-east-1"}  # registry in S3

        # node A registers a skill + a tenant, then persists to the shared store
        a = Engram(driver=MockDriver(), store_dir=str(tmp_path / "a"), store=cfg, load_on_start=False)
        a.registry.register(Capability(name="sk", kind=CapabilityKind.SKILL, description="d",
                                       routing_key=[0.1], project="default", eval_passed=True))
        key, tenant = a.add_tenant(name="acme")
        a.save()

        # node B starts fresh against the same store and sees everything
        b = Engram(driver=MockDriver(), store_dir=str(tmp_path / "b"), store=cfg, load_on_start=True)
        assert b.registry.get("sk") is not None
        assert b.tenants.resolve(key).project == tenant.project

        # a change on A is picked up by B via reload()
        a.registry.register(Capability(name="sk2", kind=CapabilityKind.SKILL, description="d2",
                                       routing_key=[0.2], project="default", eval_passed=True))
        a.save()
        assert b.registry.get("sk2") is None     # not yet
        b.reload()
        assert b.registry.get("sk2") is not None  # picked up after reload
