# External artifact integrity

ModSSC exposes a method-independent runtime contract for files that must stay
unchanged during an experiment. `ArtifactContract` contains only a portable
relative path, `file` or `tree`, and an expected SHA-256. The caller supplies
the machine-specific cache root separately.

```python
from modssc.runtime import (
    ArtifactContract,
    artifact_sha256,
    revalidate_artifact,
    verify_artifact,
)

digest = artifact_sha256(cache_root, path="models/encoder", kind="tree")
contract = ArtifactContract(path="models/encoder", kind="tree", sha256=digest)
preflight = verify_artifact(contract, root=cache_root)

# Immediately before use, or from another process after serialization:
revalidate_artifact(preflight, root=cache_root)
```

The benchmark runner exposes the same native contract without discovering
models or interpreting method identifiers:

```yaml
run:
  artifact_root: ${MODSSC_ARTIFACT_ROOT}
  input_artifacts:
    - path: models/encoder
      kind: tree
      sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

`artifact_root` is an operational machine path and is excluded from the
portable protocol hash. Every `input_artifacts` entry remains part of protocol
identity. The runner verifies all entries before loading the dataset, revalidates
them immediately before a successful result, and records the attestations in
`run.json`. A relative root is resolved from the YAML file's directory.

File contracts use the ordinary digest of the file bytes. Tree contracts use a
versioned canonical manifest of sorted relative paths, entry types, file sizes,
file digests, and relative symlink targets. Absolute roots and timestamps are
excluded from that digest. Verification performs a full rehash; revalidation
also compares the captured filesystem state, so content, membership, size, or
timestamp changes after preflight fail closed.
