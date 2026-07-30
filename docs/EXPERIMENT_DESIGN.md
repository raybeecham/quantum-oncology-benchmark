# Experiment Design

## Comparison principle

Every model must receive the same data partition and feature-selection result for a given repeat. Differences should arise from the model, not from hidden preprocessing advantages.

## Default demonstration design

- Stratified holdout: 75% training, 25% testing.
- Feature selection: training-only ANOVA F statistic.
- Four selected features for quantum-capable runs.
- Up to 160 stratified samples for the default quantum benchmark.
- Three repeated splits in `configs/baseline.yaml`.

These values are selected for a practical demonstration and are not optimized scientific choices.

## Why feature count is small

Angle-based feature maps generally use one qubit per selected feature. Reducing the input dimension keeps simulation and future hardware execution tractable. The same reduced feature set is provided to classical comparators to preserve fairness within the experiment.

A separate full-dimensional classical benchmark is still valuable and is provided by `configs/classical.yaml`.

## Required future enhancements

Before comparative publication:

- Nested model selection.
- Multiple feature-selection methods.
- Confidence intervals.
- Paired tests across identical folds.
- External validation.
- Learning curves.
- Computational scaling curves.
- Robustness to noise and perturbation.
