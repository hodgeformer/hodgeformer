from typing import Tuple, List

import re

from collections import deque


def _parse_layer_layout_string(layer_layout: str) -> Tuple[int, int, int]:
    """
    Parse layer structure string.
    """
    S = r"^(\d+)\:(\d+)(e?)"

    res = re.match(S, layer_layout)

    if not res:
        raise ValueError(
            (
                "Given pattern is not valid. Give a pattern of the form "
                "'\d+\:\d+e?' that is two integers separated by ':' with "
                "an optional 'e'."
            )
        )

    h, b, e = res.groups()

    h = int(h)
    b = int(b)
    e = 1 if e else 0

    if h == 0 and b == 0:
        raise ValueError(
            (
                "Given pattern is not valid. At least one of the two integer "
                "parts must be nonzero"
            )
        )

    return h, b, e


def _handle_layer_layout(N_h, layer_layout):
    """
    Define number of basic transformer layers based on input `layer_layout`
    and number of total `HodgeFormer` layers.

    Parameters
    ----------
    N_h : int
        Number of total `HodgeFormer` layers.
    layer_layout : str
        A string indicating how the hodgeformer and transformer layers are
        interleaved.
    """
    h, b, e = layer_layout

    if h == 0:
        N_b = b - (1 - e)
        N_h = 0
    else:
        N_b = b * (N_h // h - (1 - e) * (N_h % h == 0))

    return N_h, N_b


def _define_layer_layout_condition(layer_layout):
    """
    Define layer layout condition.

    DEPRECATED: NOT USED
    """
    h, b, _ = layer_layout

    if h == 0:
        cond = lambda N_b, idx: (idx <= (N_b - 1), idx)

    elif h == 1 and b == 0:
        cond = lambda N_b, idx: (False, idx)

    else:
        cond = lambda N_b, idx: (
            idx % h == (h - 1) and (idx // h) <= (N_b - 1),
            idx // h,
        )

    return cond


def interweave_two_lists(h_layers: List, b_layers: List, h: int, b: int) -> List:
    """
    Interweave two lists with a given pattern.
    """
    from collections import deque

    result = []

    h_l, b_l = deque(h_layers), deque(b_layers)

    h_cntr, b_cntr = 0, 0

    while h_l and b_l:

        if h_cntr // h == 0:
            result.append(h_l.popleft())
            h_cntr += 1

        elif h_cntr // h == 1 and b_cntr // b == 0:
            result.append(b_l.popleft())
            b_cntr += 1

        elif h_cntr // h == 1 and b_cntr // b == 1:
            h_cntr, b_cntr = 0, 0

    rest = _leftover(h_l, b_l)

    if rest:
        result.extend(rest)

    return result


def _leftover(x, y):
    """
    Return remaining list if any.
    """
    if x and not y:
        return x

    elif y and not x:
        return y

    return None


def test_layerlayout(N_h: int, layer_layout: str) -> List:
    """
    Test layer layout.
    """
    layer_layout = _parse_layer_layout_string(layer_layout)

    N_h, N_b = _handle_layer_layout(N_h, layer_layout)

    h_layers = [("H", i) for i in range(N_h)]
    b_layers = [("B", i) for i in range(N_b)]

    h, b, _ = layer_layout

    return interweave_two_lists(h_layers, b_layers, h, b)


def _test_various_layerlayout_cases():
    """
    Test cases for layer-layout.
    """
    assert test_layerlayout(1, "1:0") == [("H", 0)]
    assert test_layerlayout(1, "1:0e") == [("H", 0)]
    assert test_layerlayout(2, "1:0") == [("H", 0), ("H", 1)]
    assert test_layerlayout(2, "1:0e") == [("H", 0), ("H", 1)]

    assert test_layerlayout(0, "0:1") == []
    assert test_layerlayout(0, "0:1e") == [("B", 0)]
    assert test_layerlayout(0, "0:2e") == [("B", 0), ("B", 1)]

    assert test_layerlayout(1, "1:1") == [("H", 0)]
    assert test_layerlayout(1, "1:1e") == [("H", 0), ("B", 0)]

    assert test_layerlayout(2, "2:1") == [("H", 0), ("H", 1)]
    assert test_layerlayout(2, "2:1e") == [("H", 0), ("H", 1), ("B", 0)]

    assert test_layerlayout(2, "2:2") == [("H", 0), ("H", 1)]
    assert test_layerlayout(2, "2:2e") == [("H", 0), ("H", 1), ("B", 0), ("B", 1)]

    assert test_layerlayout(2, "2:2") == [("H", 0), ("H", 1)]
    assert test_layerlayout(2, "2:2e") == [("H", 0), ("H", 1), ("B", 0), ("B", 1)]

    assert test_layerlayout(2, "1:2") == [("H", 0), ("B", 0), ("B", 1), ("H", 1)]
    assert test_layerlayout(2, "1:2e") == [
        ("H", 0),
        ("B", 0),
        ("B", 1),
        ("H", 1),
        ("B", 2),
        ("B", 3),
    ]

    assert test_layerlayout(3, "3:1e") == [("H", 0), ("H", 1), ("H", 2), ("B", 0)]
    assert test_layerlayout(3, "3:2e") == [
        ("H", 0),
        ("H", 1),
        ("H", 2),
        ("B", 0),
        ("B", 1),
    ]
