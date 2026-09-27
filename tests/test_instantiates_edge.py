from pathlib import Path
from axon.core.graph.model import RelType
from axon.core.ingestion.pipeline import run_pipeline


def test_js_instantiates_edge(tmp_path: Path) -> None:
    a_js = tmp_path / "a.js"
    b_js = tmp_path / "b.js"

    a_js.write_text(
        "import { UserService } from './b';\nconst u = new UserService();\n",
        encoding="utf-8",
    )
    b_js.write_text("export class UserService {}\n", encoding="utf-8")

    graph, _ = run_pipeline(tmp_path, embeddings=False)

    instantiates_rels = [
        rel for rel in graph.iter_relationships() if rel.type == RelType.INSTANTIATES
    ]

    assert len(instantiates_rels) == 1
    rel = instantiates_rels[0]

    target_node = graph.get_node(rel.target)
    assert target_node is not None
    assert target_node.name == "UserService"
    assert target_node.file_path == "b.js"

    assert rel.properties.get("line") == 2


def test_python_instantiates_edge(tmp_path: Path) -> None:
    py_a = tmp_path / "app.py"
    py_b = tmp_path / "service.py"

    py_a.write_text("from service import Service\ns = Service()\n", encoding="utf-8")
    py_b.write_text("class Service:\n    pass\n", encoding="utf-8")

    graph, _ = run_pipeline(tmp_path, embeddings=False)

    instantiates_rels = [
        rel for rel in graph.iter_relationships() if rel.type == RelType.INSTANTIATES
    ]

    assert len(instantiates_rels) == 1
    rel = instantiates_rels[0]

    target_node = graph.get_node(rel.target)
    assert target_node is not None
    assert target_node.name == "Service"
    assert target_node.file_path == "service.py"

    assert rel.properties.get("line") == 2


def test_factory_function_no_instantiation(tmp_path: Path) -> None:
    """Test A: Call to Router() without 'new' in JS should NOT emit INSTANTIATES edge."""
    a_js = tmp_path / "a.js"
    b_js = tmp_path / "b.js"

    b_js.write_text("export class Router {}\n", encoding="utf-8")
    a_js.write_text(
        "import { Router } from './b';\nRouter();\n",
        encoding="utf-8",
    )

    graph, _ = run_pipeline(tmp_path, embeddings=False)

    instantiates_rels = [
        rel for rel in graph.iter_relationships() if rel.type == RelType.INSTANTIATES
    ]
    assert len(instantiates_rels) == 0


def test_python_isinstance_no_instantiation(tmp_path: Path) -> None:
    """Test B: Python isinstance(x, MyClass) should NOT emit INSTANTIATES edge."""
    py_a = tmp_path / "a.py"
    py_b = tmp_path / "b.py"

    py_b.write_text("class MyClass:\n    pass\n", encoding="utf-8")
    py_a.write_text("from b import MyClass\nx = 42\nisinstance(x, MyClass)\n", encoding="utf-8")

    graph, _ = run_pipeline(tmp_path, embeddings=False)

    instantiates_rels = [
        rel for rel in graph.iter_relationships() if rel.type == RelType.INSTANTIATES
    ]
    assert len(instantiates_rels) == 0


def test_generic_constructor_no_instantiation_for_type_arg(tmp_path: Path) -> None:
    """Test C: new Map<string, Foo>() should NOT emit INSTANTIATES edge targeting Foo."""
    a_ts = tmp_path / "a.ts"
    b_ts = tmp_path / "b.ts"

    b_ts.write_text("export class Foo {}\n", encoding="utf-8")
    a_ts.write_text(
        "import { Foo } from './b';\nconst m = new Map<string, Foo>();\n",
        encoding="utf-8",
    )

    graph, _ = run_pipeline(tmp_path, embeddings=False)

    foo_instantiates = [
        rel for rel in graph.iter_relationships()
        if rel.type == RelType.INSTANTIATES
        and (target := graph.get_node(rel.target)) is not None
        and target.name == "Foo"
    ]
    assert len(foo_instantiates) == 0
