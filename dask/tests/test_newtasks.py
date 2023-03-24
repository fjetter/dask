import dask
from dask.newtasks import Key, NewSubgraphCallable, Task, convert_graph, get_from_tasks


def test_translate_graphs():
    def concat(*args):
        return "".join(args)

    def concat2(*args):
        return ".".join(args)

    dsk = {
        "key-1": (concat, "a", "b"),
        "key-2": (concat, "key-1", "c"),
        "key-3": (concat, (concat2, "d", "key-1"), "key-2"),
    }

    converted = convert_graph(dsk)

    expected = {
        "key-1": Task(
            "key-1",
            concat,
            ("a", "b"),
        ),
        "key-2": Task(
            "key-2",
            concat,
            (Key("key-1"), "c"),
        ),
        "key-3": NewSubgraphCallable(
            {
                "_anon-0": Task("_anon-0", concat2, ("d", Key("key-1"))),
                "key-3": Task(
                    "key-3",
                    concat,
                    (
                        Key("_anon-0"),
                        Key("key-2"),
                    ),
                ),
            },
            outkey="key-3",
            inkeys=["key-1", "key-2"],
        ),
    }

    assert expected["key-1"] == converted["key-1"]
    assert expected["key-2"] == converted["key-2"]
    # assert expected['key-3'] == converted['key-3'] # Different names
    # assert expected == converted

    key_1 = converted["key-1"]()
    key_2 = converted["key-2"](key_1)
    key_3 = converted["key-3"](key_1, key_2)
    assert (key_3,) == get_from_tasks(converted, ("key-3",))
    assert (key_3,) == dask.get(dsk, keys=["key-3"])
