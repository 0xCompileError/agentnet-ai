# Topology Search

Run bounded topology search directly from ordinary training inputs and expected
outputs. The example uses `an.train(..., optimize=TopologyOptimizer(...))`, so
no compiled-graph scorer or execution adapter is required.

```bash
PYTHONPATH=src:. python examples/10_topology_search/main.py
```
