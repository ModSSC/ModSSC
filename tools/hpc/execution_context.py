from __future__ import annotations

import os
from collections.abc import Mapping

_EXECUTION_ENVIRONMENT = {
    "job_id": "MODSSC_EXECUTION_JOB_ID",
    "job_name": "MODSSC_EXECUTION_JOB_NAME",
    "array_job_id": "MODSSC_EXECUTION_ARRAY_JOB_ID",
    "array_task_id": "MODSSC_EXECUTION_ARRAY_TASK_ID",
    "cluster_name": "MODSSC_EXECUTION_CLUSTER",
    "constraints": "MODSSC_EXECUTION_CONSTRAINTS",
    "partition": "MODSSC_EXECUTION_PARTITION",
    "qos": "MODSSC_EXECUTION_QOS",
    "node_name": "MODSSC_EXECUTION_NODE",
}


def execution_metadata(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return operational scheduler metadata injected by the HPC adapter."""

    env = os.environ if environ is None else environ
    return {
        field: value
        for field, variable in _EXECUTION_ENVIRONMENT.items()
        if (value := env.get(variable))
    }


def is_scheduled_execution(environ: Mapping[str, str] | None = None) -> bool:
    """Whether a scheduler adapter authenticated the current allocation."""

    env = os.environ if environ is None else environ
    return bool(env.get("MODSSC_EXECUTION_JOB_ID"))
