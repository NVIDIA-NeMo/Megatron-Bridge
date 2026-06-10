# Stacked PR hello-world

A throwaway document used to demonstrate the `gh stack` workflow. Each layer
of the stack appends one section, so every PR in the chain shows a small,
self-contained diff.

## Layer 1 — hello world

This is the bottom of the stack. It targets `main`.

## Layer 2 — how stacking works

This layer is built on top of layer 1. Its PR targets the layer-1 branch, so
the diff GitHub shows is only this section — reviewers never re-review layer 1.

## Layer 3 — teardown

Top of the stack. Once the demo is done, the whole chain (branches + PRs) is
removed in one command: `gh stack unstack`.
