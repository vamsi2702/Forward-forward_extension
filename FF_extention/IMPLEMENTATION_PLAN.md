# Forward-Forward Extension: Implementation Plan

## Scope
- Objective 1: Compare alternative goodness functions (10 to 20 variants).
- Objective 2: Compare alternative activation functions (10 to 20 variants).
- Objective 3: Implement local goodness functions for spatial data.
- Stretch goal (sleep phase) intentionally excluded for this milestone.

## Code Structure
- `src/ff_research/goodness.py`: Goodness function registry and normalization policies.
- `src/ff_research/activations.py`: Activation registry including Student-t negative log-density activation.
- `src/ff_research/models.py`: Fully connected FF layers and locally connected spatial FF layers.
- `src/ff_research/trainers.py`: Layer-wise FF training loops and label-search evaluation.
- `src/ff_research/experiments.py`: Sweep runners for Objectives 1 and 2 plus spatial experiments for Objective 3.
- `notebooks/01_objective_goodness_functions.ipynb`: Objective 1 experiments.
- `notebooks/02_objective_activation_functions.ipynb`: Objective 2 experiments.
- `notebooks/03_objective_local_spatial_goodness.ipynb`: Objective 3 experiments.

## Experimental Defaults
- Dataset: MNIST for Objectives 1 and 2.
- Objective 3: Jittered MNIST with locally connected, no-weight-sharing FF layers.
- Evaluation: Label-conditioned goodness search across all classes.

## Objective Coverage
- Objective 1 Goodness alternatives: 14 functions.
- Objective 2 Activation alternatives: 17 functions.
- Objective 3 Spatial local goodness: configurable local patch block architecture and ablation-ready notebook cells.

## Notes
- Full sweeps can be compute intensive. Each notebook contains a quick smoke-test cell and a full experiment cell.
- The implementation follows the FF note that unsquared activity objectives should use L1-style normalization.
