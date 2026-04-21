"""Shared labels and schema constants for the numerical core."""

CONWAY_SUBGROUP_FLAGS = {
    "icu": "is_icu",
    "arf": "is_arf",
    "lft": "is_lft",
}

PACO2_REQUIRED_COLUMNS = {"paco2", "is_amb", "is_emer", "is_inp", "cc_time"}
PACO2_SUBGROUP_ORDER = ("pft", "ed_inp", "icu")
PACO2_PRIOR_GROUPS = ("pft", "ed_inp", "icu", "all")
PACO2_PRIOR_REQUIRED_COLUMNS = {"group", "paco2_bin", "weight"}
DEFAULT_PACO2_QUANTILES: tuple[float, ...] = (0.025, 0.25, 0.5, 0.75, 0.975)
