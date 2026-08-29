# Calder MNIST VAE2 reconstruction v1

## Scope

This execution covers the ten MNIST Table 1 cards for Laplace Learning and
Poisson Learning, from one through five labels per class and 100 sampling seeds
per cell: 1,000 declared runs in total.

The environment and authenticated MNIST cache are prepared before scheduling.
Each Slurm task then invokes only the public interface:

```bash
modssc-bench --config CARD --seed-index N
```

The selected YAML activates the complete native scientific chain: load MNIST,
draw labels, train or reuse VAE2, build or reuse the Annoy graph, run the method,
evaluate, and write `run.json`. No author embedding, neighbour bank, graph, or
label permutation is an input.

## Shared construction

All ten cards deliberately declare the same fixed preprocessing and graph
identity. One cold run trains VAE2 and builds the graph; later runs reuse the
authenticated caches. Only the sampling seed changes. Laplace and Poisson read
the same graph, with Poisson removing self-loops inside its native paper solver.

The cluster adapter is private and generic. It selects the V100 allocation,
isolated source/environment, cache/output roots, wall time, and array throttle;
it contains no article or method behaviour. Existing jobs are neither stopped
nor modified.

## Scientific claim

The maximum claim is `paper_approx`. VAE2 was produced after the article, and
the authors did not publish the exact training seed, Annoy seed/version, or 100
label draws as a complete executable protocol. The cards resolve those gaps
with fixed ModSSC seeds and recorded runtime identity while retaining the
original Table 1 targets and margins unchanged.

## Durable results

Raw per-seed reports, scheduler logs, models, graphs, and caches stay outside
Git. After all declared seeds are reconciled, the compact immutable publication
bundle belongs at:

```text
docs/replications/results/paper/calder-mnist-vae2-v1/
```

That bundle will contain a human summary, source/environment and card digests,
one compact aggregate per card, explicit failed or missing seeds, an external
raw-archive digest, and `SHA256SUMS`. It will be created from fresh results only;
this design page is not an execution result.
