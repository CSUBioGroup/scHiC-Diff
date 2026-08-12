"""Central method identities, display labels, orders, and plot styles."""

from dataclasses import dataclass


ALL_METHODS = (
    "raw",
    "schicluster",
    "higashi_nbr0",
    "higashi_nbr5",
    "flamingo",
    "scvi3d",
    "schicdiff",
)
IMPUTED_METHODS = ALL_METHODS[1:]
MAIN_UMAP_METHODS = ("raw", "schicluster", "higashi_nbr5", "schicdiff")
MAIN_CONTACT_METHODS = (
    "raw",
    "schicluster",
    "higashi_nbr5",
    "scvi3d",
    "schicdiff",
)
MAIN_APA_METHODS = ("schicluster", "higashi_nbr5", "scvi3d", "schicdiff")

# Compatibility alias for archived scripts that used one shared main selection.
MAIN_METHODS = MAIN_UMAP_METHODS

STAGES = ("E7.0", "E7.5", "E8.0", "E8.5", "E9.5", "E10.5", "E11.5")
MAIN_CELL_COUNTS = (10, 100, 476)
ALL_CELL_COUNTS = (10, 100, 200, 476)
APA_TOP_N_VALUES = (10, 20, 50)
TOP_N_VALUES = (10, 20, 50, 100, 200)
SEEDS = (42, 43, 44)

DISPLAY_LABELS = {
    "raw": "Raw",
    "schicluster": "scHiCluster",
    "higashi_nbr0": "Higashi-nbr0",
    "higashi_nbr5": "Higashi-nbr5",
    "flamingo": "FLAMINGO",
    "scvi3d": "scVI-3D",
    "schicdiff": "scHiC-Diff",
}

STAGE_STORAGE_TO_DISPLAY = {
    "E70": "E7.0",
    "E75": "E7.5",
    "E80": "E8.0",
    "E85": "E8.5",
    "E95": "E9.5",
    "EX05": "E10.5",
    "EX15": "E11.5",
}


@dataclass(frozen=True)
class MethodStyle:
    color: str
    marker: str
    linestyle: str = "-"
    filled: bool = False


METHOD_STYLES = {
    "schicluster": MethodStyle("#CC79A7", "v"),
    "higashi_nbr0": MethodStyle("#56B4E9", "P"),
    "higashi_nbr5": MethodStyle("#E69F00", "X"),
    "flamingo": MethodStyle("#0072B2", "s"),
    "scvi3d": MethodStyle("#009E73", "^"),
    "schicdiff": MethodStyle("#D55E00", "D", filled=True),
}

GROUP_STYLES = {
    "Red": {"color": "#D55E00", "marker": "o"},
    "Blue": {"color": "#0072B2", "marker": "^"},
}


def _alias_key(value):
    return str(value).strip().lower().replace("–", "-").replace("—", "-")


_ALIASES = {
    "raw": "raw",
    "raw sc-hi-c": "raw",
    "schicluster": "schicluster",
    "higashi-0": "higashi_nbr0",
    "higashi nbr0": "higashi_nbr0",
    "higashi_nbr0": "higashi_nbr0",
    "higashi_0": "higashi_nbr0",
    "higashi-5": "higashi_nbr5",
    "higashi nbr5": "higashi_nbr5",
    "higashi_nbr5": "higashi_nbr5",
    "higashi_5": "higashi_nbr5",
    "flamingo": "flamingo",
    "scvi-3d": "scvi3d",
    "scvi3d": "scvi3d",
    "baseline_schicdiff": "schicdiff",
    "schicdiff": "schicdiff",
    "schic-diff": "schicdiff",
}


def canonical_method(value):
    """Return the canonical method ID for historical storage/display aliases."""

    key = _alias_key(value)
    try:
        return _ALIASES[key]
    except KeyError:
        raise ValueError("unknown method: {!r}".format(value))


def display_label(value):
    """Return the final standardized display label for a method alias."""

    return DISPLAY_LABELS[canonical_method(value)]
