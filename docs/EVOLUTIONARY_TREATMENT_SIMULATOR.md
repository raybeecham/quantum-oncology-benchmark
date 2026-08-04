# Evolutionary Treatment Simulator

## Purpose

`evolution-protocol-v1` introduces an assumption-driven computational model of tumor evolution under treatment pressure. It represents a tumor as two competing populations:

- a treatment-sensitive population,
- a treatment-resistant population.

The protocol is designed to make the biological assumptions, treatment rules, solver behavior, and interpretation boundaries explicit. It is an educational research simulator, not a patient-specific digital twin or clinical decision-support system.

## Command

```bash
qob evolve --config configs/evolution-two-clone.yaml
```

An output directory can be overridden without changing the profile:

```bash
qob evolve \
  --config configs/evolution-two-clone.yaml \
  --output reports/evolution-two-clone-run-2
```

## Model

The deterministic two-clone model is:

```text
dS/dt = rS*S*(1 - (S + alpha_SR*R)/K) - killS*u(t)*S - mu*S
dR/dt = rR*R*(1 - (R + alpha_RS*S)/K) - killR*u(t)*R + mu*S
```

Where:

- `S` is the sensitive population,
- `R` is the resistant population,
- `rS` and `rR` are growth rates,
- `K` is carrying capacity,
- `alpha_SR` is the competitive effect of resistant cells on sensitive cells,
- `alpha_RS` is the competitive effect of sensitive cells on resistant cells,
- `killS` and `killR` are treatment-associated kill rates,
- `u(t)` is treatment intensity,
- `mu` is an optional one-way sensitive-to-resistant transition rate.

The reference profile sets `mu` to zero so the first experiment isolates selection of a pre-existing resistant population. Acquired transition sensitivity should be evaluated as a separate versioned profile rather than silently modifying the reference assumptions.

## Treatment policies

### No treatment

`treatment_intensity = 0` for the full horizon. This is an ecological growth control, not a clinical recommendation.

### Continuous treatment

`treatment_intensity = 1` for the full horizon. It provides the strongest persistent selection pressure in the reference model.

### Fixed intermittent treatment

Treatment alternates between fixed on and off durations. The reference profile uses 14 days on and 14 days off.

### Burden-adaptive treatment

Treatment starts on, stops when total modeled burden falls below a configured fraction of initial burden, and restarts when burden rises to the restart threshold. The reference profile uses:

- stop at 50% of initial burden,
- restart at 100% of initial burden.

These thresholds are model inputs. They are not treatment guidance.

## Numerical execution

The simulator uses SciPy `solve_ivp` over piecewise intervals. Treatment intensity remains fixed within each interval and may change at the next configured time step. The reference profile uses a one-day policy update interval.

The execution is deterministic for a fixed profile because `evolution-protocol-v1` contains no stochastic process. A future protocol may add branching-process or Gillespie-style stochastic evolution as a separate model family.

## Metrics

Each strategy records:

- final, minimum, and maximum total burden,
- day and value of tumor-burden nadir,
- final and maximum resistant fraction,
- first resistant-dominance threshold crossing,
- first configured progression-threshold crossing,
- tumor-burden area under the curve,
- average two-clone Shannon diversity,
- time below initial modeled burden,
- cumulative dose-days,
- fraction of time on treatment,
- treatment cycle count.

These are simulation metrics. They do not establish efficacy, toxicity, survival benefit, or clinical utility.

## Artifact package

Each run writes:

- `evolution_experiment.json`: complete configuration, model declaration, trajectories, schedules, summaries, events, environment, and simulation fingerprint.
- `population_trajectories.csv`: time-resolved sensitive, resistant, total-burden, clonal-fraction, and diversity values.
- `treatment_schedule.csv`: one row per treatment-policy interval.
- `strategy_summary.csv`: one aggregate row per strategy.
- `evolutionary_events.csv`: treatment transitions, burden nadirs, resistance thresholds, and progression thresholds.
- `EVOLUTION_REPORT.md`: bounded human-readable report.

## Interpretation boundaries

This first protocol intentionally excludes:

- patient calibration,
- drug-specific pharmacokinetics or pharmacodynamics,
- toxicity and dose-limiting adverse events,
- immune dynamics,
- vascular or spatial structure,
- metastatic compartments,
- reversible drug-tolerant states,
- multi-drug interactions,
- stochastic mutation and extinction,
- clinical outcome prediction.

A strategy that performs better under one parameter profile may perform worse under another. The result is always conditional on the declared model assumptions.

## Roadmap

The intended progression is:

1. deterministic two-clone selection,
2. parameter-sensitivity and virtual-patient cohorts,
3. stochastic mutation and clonal extinction,
4. reversible drug-tolerant states,
5. multi-clone and multi-drug models,
6. spatial or agent-based tumor ecosystems,
7. classical treatment-policy optimization,
8. matched quantum or hybrid optimization experiments.

Quantum methods are not inserted into the biological dynamics by default. They must compete against strong classical optimizers over the same declared treatment-policy search space.
