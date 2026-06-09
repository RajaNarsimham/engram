import pytest

from engram.persistence import make_store
from engram.persistence.store import FileStore


def test_make_store_defaults_to_file(monkeypatch, tmp_path):
    monkeypatch.delenv("ENGRAM_S3_BUCKET", raising=False)
    assert isinstance(make_store(None, work_dir=str(tmp_path)), FileStore)


def test_make_store_dict_file(tmp_path):
    assert isinstance(make_store({"type": "file", "root": str(tmp_path)}), FileStore)


def test_make_store_env_selects_aws(monkeypatch, tmp_path):
    pytest.importorskip("boto3")
    monkeypatch.setenv("ENGRAM_S3_BUCKET", "some-bucket")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    s = make_store(None, work_dir=str(tmp_path))
    assert type(s).__name__ == "AWSStore"


def test_filestore_registry_roundtrip(tmp_path):
    s = FileStore(str(tmp_path))
    recs = [{"name": "a", "project": "default", "kind": "skill", "routing_key": [0.1, 0.2]}]
    s.push_registry(recs)
    assert s.pull_registry() == recs


def test_filestore_dir_roundtrip(tmp_path):
    s = FileStore(str(tmp_path / "store"))
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.txt").write_text("hello")
    s.push_dir("blobs/x", str(src))
    dest = tmp_path / "out"
    assert s.pull_dir("blobs/x", str(dest)) is True
    assert (dest / "f.txt").read_text() == "hello"
    assert s.list_dirs("blobs") == ["x"]


def test_aws_store_roundtrip(tmp_path):
    pytest.importorskip("moto")
    import boto3
    from moto import mock_aws

    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="engram-test-bucket")
        boto3.resource("dynamodb", region_name="us-east-1").create_table(
            TableName="engram-registry", KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "pk", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST")
        s = make_store({"type": "aws", "bucket": "engram-test-bucket", "table": "engram-registry",
                        "region": "us-east-1"}, work_dir=str(tmp_path))
        recs = [{"name": "a", "project": "default", "kind": "skill", "routing_key": [0.1, 0.2]}]
        s.push_registry(recs)
        assert s.pull_registry() == recs
        src = tmp_path / "src"
        src.mkdir()
        (src / "f.bin").write_bytes(b"xyz")
        s.push_dir("projects/p", str(src))
        out = tmp_path / "out"
        assert s.pull_dir("projects/p", str(out)) is True
        assert (out / "f.bin").read_bytes() == b"xyz"
        assert s.list_dirs("projects") == ["p"]
