from axon.core.parsers.base import CallInfo
from axon.core.parsers.typescript import TypeScriptParser


def test_new_expression_flag_true() -> None:
    parser = TypeScriptParser(dialect="typescript")
    code = """\
const service = new UserService();
const foo = new ns.Foo();
"""
    result = parser.parse(code, "test.ts")

    new_calls = [c for c in result.calls if c.is_new]
    assert len(new_calls) == 2

    user_service_call = next(c for c in result.calls if c.name == "UserService")
    assert user_service_call.is_new is True
    assert user_service_call.receiver == ""

    foo_call = next(c for c in result.calls if c.name == "Foo")
    assert foo_call.is_new is True
    assert foo_call.receiver == "ns"


def test_regular_call_flag_false() -> None:
    parser = TypeScriptParser(dialect="typescript")
    code = """\
someFunction();
"""
    result = parser.parse(code, "test.ts")

    assert len(result.calls) == 1
    call = result.calls[0]
    assert call.name == "someFunction"
    assert call.is_new is False


def test_call_info_default_is_new() -> None:
    call = CallInfo(name="test", line=1)
    assert call.is_new is False
