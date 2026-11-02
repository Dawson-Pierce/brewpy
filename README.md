# brewpy

Python bindings for the BREW C++ tracking library. The binding generator and
handwritten bindings preserve the local implementation, including unpublished
working-tree updates. They are generated from the same `@mex`-annotated BREW
headers as the MATLAB bindings.

## Public API

The Python API mirrors the C++ roles: `models`, `dynamics`, `filters`,
`multiObjectTrackers`, and `utils`. `multiObjectTrackers.PHD(filter)` infers the
density from a configured native filter. For empty construction or density-specific
setup, use its named native specialization, such as `TemplatePosePHD` or `USGGIWPHD`.
These are bindings of the same templated algorithm, not extra tracker classes.
Registration helpers live in `utils.registration`, clustering in `utils.clustering`,
and report fusion in `utils.fusion`. Old application/package namespaces are removed.

## Dependency layout

```text
brewpy/
  brew/                    BREW Git submodule
  brewpy/__init__.py       Python package
  generator_files/        Binding generator and handwritten bindings
  src/spherical_tracking.cpp  Spherical setup and fusion bindings
  CMakeLists.txt
  requirements.txt
```

BREW is supplied by the nested `brew/` submodule. After cloning, initialize
nested dependencies:

```sh
git submodule update --init --recursive
```

## Build

The spherical tracking API includes `multiObjectTrackers.SphericalPHDConfig`,
`multiObjectTrackers.USGGIWPHD`, `utils.fusion.ReportFusion`, and
`multiObjectTrackers.TrajectoryUSGGIWPHD`. Estimation and report fusion
execute in BREW C++.
The Python layer only supplies configuration and measurements and reads results.
Longitude/latitude inputs are N-by-2 arrays in degrees; time and angular rates
must use consistent units (the Storm application uses minutes). Filter process
covariances use radians, while configuration fields ending in `_deg` use degrees.
`ReportFusion.fuse_rings` accepts lists of mixtures in routing order and returns
the reduced ring mixtures and pooled ground mixture. Empty rings are allowed.

Construct `multiObjectTrackers.USGGIWPHD(config)` or
`multiObjectTrackers.TrajectoryUSGGIWPHD(config, ...)` to configure
the complete measurement-driven `step`. The existing no-argument constructors
and explicit `set_filter`, `predict`, `correct`, and `cleanup` API remain available.
Read posterior snapshots with `phd.latest()` directly. There is no nested `.phd`
object or separate `Tracker` class. Trajectory `expected_cardinality()` sums the
full posterior; `extracted_cardinality()` sums only extracted components.

Named Python RFS specializations live under `multiObjectTrackers` and bind the shared C++ algorithms:
`brew::multiObjectTrackers::PHD<Density>`, `CPHD`, `GLMB`, `JGLMB`, `MBM`, and `PMBM`. They introduce no
extra estimator object. Python also accepts a configured compatible filter
directly, for example `phd = brewpy.multiObjectTrackers.PHD(filter)`.

Attach optional intent inference with an outer native filter:

```python
# Configure the primary filter and intent estimator first.
combined = brewpy.filters.IntentFilter(primary_filter, intent_estimator)
phd = brewpy.multiObjectTrackers.PHD(combined)
```

`brewpy.filters.IntentFilter` is overloaded for the bound Gaussian, GGIW, USGGIW,
trajectory GGIW/USGGIW, and template-pose filter families. The estimator's behavior
models must match the primary density type. The existing RFS constructors accept
the wrapped filter directly; no additional tracker or augmented RFS class is needed.
`density.intent` exposes the optional `brewpy.models.CategoricalIntent` state
for every family. Spacecraft firing reports are separate physical-filter results:
`density.firing` exposes `models.FiringEstimate`, including without an HMM.
`filters.IntentEstimator(hypotheses)` and `models.IntentHypothesis(name, model)`
select the matching native density specialization. For example,
`models.GaussianUninformativeBehavior()` supplies zero log evidence for a Gaussian
hypothesis with no observation model. Additional informative models implement the native C++ interface.

`IntentFilter` owns a clone of the primary filter and a copy of the HMM configuration.
Each density carries its own intent posterior and bounded previous-observation
context. Prediction delegates geometric propagation and applies the HMM transition
once. Correction delegates to the primary filter, then applies Bayes to that
association branch's intent probabilities. Returned likelihoods and RFS weights
are unchanged. Missed detections retain the predicted probabilities, with no
invented observation evidence. Gating and branching remain primary-filter operations.

Some primary filters finish paired firing evidence in `adjust_log_weights`.
After that operation the wrapper refreshes intent from the saved HMM prior, so
neither the transition nor the evidence is counted twice. Cleanup only reduces
existing densities; it does not run another HMM or associate object reports.
`TmEkf` itself contains no intent estimation. Configure it before wrapping.

Geometric merging ignores intent labels and probabilities. It marginalizes intent
by intensity weight, aligning labels by name, and combines firing moments and
distribution components. Merged firing intervals come from the mixture CDF.
Under the categorical assumption, `p_merged = sum(w_i*p_i)/sum(w_i)`; an unmarked
birth supplies a uniform prior. This preserves the current categorical marginal,
not correlations with the pre-merge state histories. The next prediction captures
the merged primary density. Mixtures expose `merge(threshold)` for the same native
reduction used by the RFS.
Tracking results are invariant to this downstream computation; intent probabilities
can differ from the former strongest-report, post-reduction calculation.
The RFS algorithm contains no
template-pose or spherical estimation equations; those belong to the
single-object filters. See `brew/README.md` for the C++ construction contract.

`IntentEstimator` requires application-supplied `IntentHypothesis(name, model)`
values. Each model implements the native `PredictiveBehaviorModel` interface.
The dissertation uses `PointingFiringModel`; the HMM does not inspect its type,
pointing rule, or label:

```python
from brewpy import models, dynamics, filters, multiObjectTrackers

impulse = models.ImpulsePrior(maximum=0.2, probability=0.01)
# Configure the primary pose filter before wrapping it:
motion = dynamics.ClohessyWiltshire(mean_motion)
motion.set_impulse_model(dynamics.ImpulseModel(impulse, body_frame=True))
primary.set_dynamics(motion)
hypotheses = [
    models.IntentHypothesis("coast", models.PointingFiringModel(
        models.PointingRule.Fixed, impulse, firing_probability=0.0)),
    models.IntentHypothesis("approach", models.PointingFiringModel(
        models.PointingRule.Radial, impulse, radial_sign=-1)),
]
estimator = filters.IntentEstimator(hypotheses, dt=30.)
```

For `PointingFiringModel`, `Fixed` normalizes `axis`; `Radial` points along relative position with sign
`+1` or `-1`; `Tangential` normalizes `axis × radial`. The journal supplies these
values once. Hypothesis-specific firing probabilities select mixtures of no-burn
and positive-burn evidence. Define `ImpulsePrior` once and pass it to the primary
motion model and behavior models. It supplies the positive-impulse bound
and physical firing prior. Models inherit that probability unless explicitly
overridden (coast uses zero). `TmEkf` computes the separate firing report using
its own prior and evidence. `IntentEstimator` has no impulse parameters or
firing-posterior methods; it only evaluates hypotheses and performs the HMM.
Consumers copy this immutable setup; configure them before wrapping the filter.
Likelihood evaluation, integration over gain/impulse magnitude, and HMM updates
execute in C++. Arbitrary labels and hypothesis counts are supported.

Pointing uncertainty, gain/holding priors, and evidence-ablation switches now
belong to `PointingFiringModel`, not `IntentEstimator`. The estimator retains the
timestep and HMM persistence. The spacecraft estimator instantiates the shared C++
`filters::IntentEstimator<models::TemplatePose<>>`; it has no second association tracker.
Model diagnostics (`branch_log_likelihoods` and
`impulse_log_likelihood`) are available on `hypothesis.model`.

In C++, a hypothesis also accepts a named native log-likelihood function directly;
no helper factory or subclass is needed for that case (see `brew/README.md`).
Python setup selects compiled native models; it does not execute Python callbacks.
To expose a different native behavior family to Python, implement `validate()` and
`log_evidence(current, previous, dt)` in a C++ subclass of
`PredictiveBehaviorModel`, then expose its constructor through brewpy. The method
owns prediction and likelihood evaluation using the full current/previous pose
densities (state, covariance, attitude, and existing firing evidence). Return one
joint log likelihood for coupled observations, or ordered contributions for a
justified factorization. The HMM adds them to its predicted log probability.
The existing model returns the original direction and firing contributions in
their original order. `posterior_history()` evaluates a known object's density
sequence in C++; `evidence_recursion()` evaluates a supplied log-evidence matrix.
Both methods belong to the shared HMM and are available for the other bound
density families too. Plotting reads saved journal records directly. The former
`step()`, `reports()`, `reset()`, association `gate`, `IntentPose`, and
`IntentEstimate` compatibility API has been removed; rerun simulation when old
results lack per-object intent posteriors.

Models are shared immutable configuration. Do not keep mutable per-object state
inside them: the existing density/history owns object-specific state. A null
`previous` means consecutive-pose evidence is unavailable; newly born objects
still receive the original uniform intent prior before evidence updates begin.
This interface supports models conditioned on the available density pair; extra
latent trajectory state is deliberately not introduced in this change.
No Python override/trampoline is provided, so inference stays native. No trajectory
or guidance hypotheses are included. The journal attaches this estimator through
the outer `IntentFilter` described above.

Impulse prediction is owned by `dynamics.ImpulseModel`, attached through
`motion.set_impulse_model(...)`. Gaussian EKF/UKF and GGIWEKF can use the same
model with a direction in the tracking frame; `body_frame=True` uses the attitude
estimate supplied by TmEkf. The default velocity indices are `[3, 4, 5]`; configure
`axis=[1, 0], velocity_indices=[2, 3]` for a planar constant-velocity state.
Prediction branches are handled by the existing filters and RFS, without another
tracker class. TmEkf no longer exposes `set_impulse_prior`, `set_impulse_max`,
`set_impulse_probability`, or `set_thruster_axis`. Its firing inference reads the
motion model's prior. Trajectory filters and IMM require their own compatible
branching implementation; they do not silently apply a mismatched impulse model.

After building, run the binding regressions with
`python -m unittest discover -s tests`.

Use Python 3.10+, a C++20 compiler, CMake 3.20+, and installed Eigen 3.4 headers.
In a virtual environment, from this checkout:

```sh
python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
  -DPython3_EXECUTABLE="$(python -c 'import sys; print(sys.executable)')"
cmake --build build --parallel 2
python -c "import brewpy; print(brewpy.models.TemplatePose)"
```

PowerShell configuration uses the same CMake arguments on one line, or backticks
for line continuation. CMake copies the compiled extension into `brewpy/`.
To import it from elsewhere, include this repository root on `PYTHONPATH`.
Build the extension for the Python version and operating system that will run it;
compiled extensions are ignored by Git.

The application builds require installed Eigen and pybind11 and do not fetch
source code. BREW's tests, examples, and C++ plotting are disabled for the Python
build. Their source remains in the standalone BREW dependency. Both applications
use the same dependency revisions; application-specific scripts live outside
these packages.

## Constructor configuration

`filters.TmEkf` and both the typed PHD classes and `multiObjectTrackers.PHD`
factory accept keyword configuration. The following names are application inputs,
not library presets:

```python
primary = filters.TmEkf(
    dynamics=motion, template_library=library, icp=registration,
    process_noise=Q, measurement_noise=R, measurement_jacobian=H,
    rotation_process_noise=Q_rotation,
)
wrapped = filters.IntentFilter(primary, intent)
phd = multiObjectTrackers.PHD(wrapped, birth_model=births, **phd_settings)
```

Keyword names match setters without `set_`. TmEkf supports its dynamics,
registration and template library, noise/Jacobian, rotation noise, surface
likelihood and pose-observation settings. PHD supports `prob_detection`,
`prob_survive`, `clutter_rate`, `clutter_density`, `max_history`, `prune_threshold`,
`merge_threshold`, `merge_rotation_threshold`, `max_components`,
`extract_threshold`, `gate_threshold`, `cluster_object`, `birth_model`, and
`intensity`. This includes the two bound trajectory PHD specializations.
Unspecified fields retain existing defaults; `None` leaves a field unspecified.
Unknown names and incompatible types raise errors. Constructors use the native
Config and existing setters; there is no Python inference or wrapper class.
Birth/intensity mixtures are cloned. Existing setters and positional filter
construction remain supported. Journal settings and hypothesis definitions stay
in the journal. Other filter and RFS types retain their existing constructors.

The generator reads `@mex_config` field names from the native class and obtains
each field type with `decltype`; it contains no second set of numeric defaults.
