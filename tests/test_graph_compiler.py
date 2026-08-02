import agentnet as an


class NamedModule(an.Module):
    async def arun(self, input: object, context: object | None = None) -> object:
        return input


def test_graph_compiler_compiles_sequential_edges() -> None:
    first = NamedModule("first")
    second = NamedModule("second")

    compiled = an.compile_graph(an.Sequential(first, second))

    assert compiled.nodes == {"first": first, "second": second}
    assert compiled.edges == {"first": ("second",), "second": ()}
    assert compiled.entry_nodes == ("first",)
    assert compiled.output_nodes == ("second",)


def test_graph_compiler_compiles_parallel_reducer_edges() -> None:
    first = NamedModule("first")
    second = NamedModule("second")
    reducer = NamedModule("collector")

    compiled = an.compile_graph(an.Parallel(first, second, reducer=reducer))

    assert compiled.nodes == {"collector": reducer, "first": first, "second": second}
    assert compiled.edges == {
        "collector": (),
        "first": ("collector",),
        "second": ("collector",),
    }
    assert compiled.entry_nodes == ("first", "second")
    assert compiled.output_nodes == ("collector",)


def test_graph_compiler_compiles_dag_edges() -> None:
    first = NamedModule("first")
    second = NamedModule("second")
    dag = an.DAG(nodes={"first": first, "second": second}, edges={"first": ("second",)})

    compiled = an.compile_graph(dag)

    assert compiled.nodes == {"first": first, "second": second}
    assert compiled.edges == {"first": ("second",), "second": ()}
    assert compiled.entry_nodes == ("first",)
    assert compiled.output_nodes == ("second",)


def test_graph_compiler_is_exported_from_package_root() -> None:
    from agentnet.graphs import CompiledGraph, GraphCompiler, compile_graph

    assert an.CompiledGraph is CompiledGraph
    assert an.GraphCompiler is GraphCompiler
    assert an.compile_graph is compile_graph
