from engram.connectors.base import Connector, Document
from engram.connectors.files import DirectoryConnector, FileConnector, read_text, to_connector


def test_file_connector_reads_text(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# Title\nhello world")
    docs = list(FileConnector(f).documents())
    assert len(docs) == 1
    assert "hello world" in docs[0].text
    assert docs[0].metadata["name"] == "a.md"


def test_directory_connector_walks_and_filters(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.md").write_text("beta")
    (tmp_path / "c.png").write_bytes(b"\x89PNG")           # unsupported -> skipped
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.py").write_text("print('x')")                 # recursive
    docs = list(DirectoryConnector(tmp_path).documents())
    assert {d.metadata["name"] for d in docs} == {"a.txt", "b.md", "d.py"}


def test_html_is_stripped(tmp_path):
    f = tmp_path / "p.html"
    f.write_text("<html><body><p>Hello <b>there</b></p></body></html>")
    t = read_text(f)
    assert "Hello" in t and "there" in t and "<p>" not in t


def test_to_connector_dispatch(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("x")
    assert isinstance(to_connector(f), FileConnector)
    assert isinstance(to_connector(tmp_path), DirectoryConnector)

    class Custom(Connector):
        def documents(self):
            yield Document("d")

    c = Custom()
    assert to_connector(c) is c
