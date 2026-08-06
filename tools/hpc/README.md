# Optional HPC adapters

This directory is not part of the ModSSC wheel. It contains optional operational
adapters for running scheduler-neutral `bench` campaigns on Slurm installations.

- `config/` contains public templates with placeholders, never site credentials.
- `specs/` contains historical or large campaign layouts coupled to Slurm resource
  profiles.
- `slurm/` translates the generic execution contract into worker environment
  variables before invoking `bench`.

Paper protocol cards, datasets identities, fixed splits, scientific acceptance,
and method implementations do not depend on this directory. The autonomous
`modssc-reproduce` command runs from an installed wheel without `tools/`.
