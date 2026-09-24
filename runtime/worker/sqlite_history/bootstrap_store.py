"""Bootstrap-specific durable operation receipt transitions."""

from __future__ import annotations

from .operations import OperationStore


class BootstrapOperationStore(OperationStore):
    """Operation store whose terminal bootstrap receipts may be explicitly resumed.

    Generic runtime operation receipts remain immutable.  Bootstrap is a durable
    multi-workspace controller, so explicit resume may reopen only its own
    interrupted, failed, or cancelled parent receipt for reconciliation.
    """

    _TRANSITIONS = {
        **OperationStore._TRANSITIONS,
        "interrupted": frozenset({"reconciling"}),
        "failed": frozenset({"reconciling"}),
        "cancelled": frozenset({"reconciling"}),
    }
