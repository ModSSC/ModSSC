# Repository-only replication audits

This directory retains validators for immutable historical campaign evidence.
It is intentionally excluded from the ModSSC wheel and is not part of the
scientific runtime.

`calder/` contains the former local/HPC campaign builders and canary gates for
the completed Calder 2020 audit. They may inspect Git-bound or archived
provenance and may use scheduler adapters. The autonomous public reproduction
path instead uses `bench.campaign.protocols.calder.official` and
`bench.campaign.protocols.calder.artifacts`, which authenticate packaged
inputs and build derived caches from explicit local paths.
