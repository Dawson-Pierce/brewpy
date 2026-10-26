"""Generate the brewpy pybind11 module from annotated C++ headers.

    python generate_pybind.py --brew ../brew --out ../brew/bindings/bindings_generated.cpp
"""

import re
import argparse
from pathlib import Path
from dataclasses import dataclass, field as dfield
from typing import List, Tuple

# Bind the BREW models used by the dissertation figures. Orientation/intensity-weighted variants
# and trajectory variants still need package-specific binding work because their generic
# accessors are overloaded or windowed.
SKIP_MODELS = {"TrajectoryTemplatePose",
               "GGIWOrientation", "IGGIW",
               "TrajectoryGaussian", "TrajectoryGGIW", "TrajectoryGGIWOrientation",
               "TrajectoryIGGIW", "TrajectoryUSGGIW"}
SKIP_FILTERS = {"TrajectoryTmEkf"}
SKIP_RFS_MODELS = set()

# ---------------------------------------------------------------------------
# Annotation model + parser  (kept identical to generate_mex.py)
# ---------------------------------------------------------------------------
@dataclass
class Field:
    name: str
    type: str
    matlab_name: str

@dataclass
class DynamicsDef:
    name: str
    header: str
    args: List[Tuple[str, str]]
    fixed_state_dim: int = 0

@dataclass
class ModelDef:
    name: str
    header: str
    cpp_type: str
    fields: List[Field]
    is_trajectory: bool = False
    base_model: str = ""
    extract_extra: List[Field] = dfield(default_factory=list)
    create_handle_fields: List[Tuple[str, str]] = dfield(default_factory=list)
    create_mat_fields: List[Tuple[str, str]] = dfield(default_factory=list)
    create_int_fields: List[str] = dfield(default_factory=list)
    create_int_vec_fields: List[str] = dfield(default_factory=list)

@dataclass
class FilterDef:
    name: str
    header: str
    dist: str
    setters: List[Tuple[str, str]]
    handle_setters: List[Tuple[str, str]] = dfield(default_factory=list)

@dataclass
class RFSDef:
    name: str
    header: str
    params: List[Tuple[str, str, str]]
    init_methods: List[str] = dfield(default_factory=list)
    has_cardinality: bool = False
    has_track_histories: bool = False
    has_birth_weights: bool = False
    has_cluster_object: bool = False
    optional_params: List[Tuple[str, str]] = dfield(default_factory=list)

@dataclass
class ClusteringDef:
    name: str
    header: str
    args: List[Tuple[str, str]]

@dataclass
class IcpDef:
    name: str
    header: str
    namespace: str
    args: List[Tuple[str, str]]
    params: List[Tuple[str, str, str]] = dfield(default_factory=list)


def to_snake(name: str) -> str:
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()

def pkg_of(m) -> str:
    return m.header.split("/")[1]

def pluralize(name: str) -> str:
    if name.endswith("sis"):
        return name[:-2] + "es"
    if name.endswith(("s", "x", "z")):
        return name + "es"
    return name + "s"

def parse_fields(s: str) -> List[Field]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        t = part.split(":")
        out.append(Field(t[0].strip(), t[1].strip(),
                         t[2].strip() if len(t) > 2 else pluralize(t[0].strip())))
    return out

def parse_args_str(s: str) -> List[Tuple[str, str]]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        t = part.split(":")
        out.append((t[0].strip(), ":".join(x.strip() for x in t[1:])))
    return out

def parse_params_str(s: str) -> List[Tuple[str, str, str]]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        t = part.split(":")
        out.append((t[0].strip(), t[1].strip(), t[2].strip()))
    return out

def parse_mex_block(lines: List[str], start: int) -> dict:
    tags = {}
    m = re.search(r"@mex\s+(\w+)", lines[start])
    if m:
        tags["_category"] = m.group(1)
    i = start + 1
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("//"):
            break
        content = line.lstrip("/").strip()
        if not content.startswith("@mex_"):
            break
        m2 = re.match(r"@mex_(\w+)\s+(.*)", content)
        if m2:
            tags[m2.group(1)] = m2.group(2).strip()
        i += 1
    return tags

def scan_headers(root: Path):
    dynamics, models_list, filters = [], [], []
    rfs_types, clustering, icp_types = [], [], []
    for hpp in sorted(root.rglob("*.hpp")):
        header = str(hpp.relative_to(root)).replace("\\", "/")
        lines = hpp.read_text(encoding="utf-8").split("\n")
        for i, line in enumerate(lines):
            if "@mex " in line and "@mex_" not in line:
                tags = parse_mex_block(lines, i)
                cat = tags.get("_category", "")
                if cat == "dynamics":
                    dynamics.append(DynamicsDef(
                        tags["name"], header, parse_args_str(tags.get("args", "")),
                        int(tags.get("fixed_state_dim", "0"))))
                elif cat == "model":
                    name = tags["name"]
                    if name in SKIP_MODELS:
                        continue
                    if "trajectory" in tags:
                        models_list.append(ModelDef(name, header, f"models::{name}<>", [],
                                                    is_trajectory=True, base_model=tags["trajectory"]))
                    else:
                        models_list.append(ModelDef(
                            name, header, f"models::{name}<>",
                            parse_fields(tags.get("fields", "")),
                            extract_extra=parse_fields(tags.get("extract_extra", "")),
                            create_handle_fields=parse_args_str(tags.get("create_handle_fields", "")),
                            create_mat_fields=parse_args_str(tags.get("create_mat_fields", "")),
                            create_int_fields=[s.strip() for s in
                                               tags.get("create_int_fields", "").split(",") if s.strip()],
                            create_int_vec_fields=[s.strip() for s in
                                                   tags.get("create_int_vec_fields", "").split(",") if s.strip()]))
                elif cat == "filter":
                    if tags.get("name") in SKIP_FILTERS:
                        continue
                    filters.append(FilterDef(tags["name"], header, tags.get("dist", ""),
                                             parse_args_str(tags.get("setters", "")),
                                             handle_setters=parse_args_str(tags.get("handle_setters", ""))))
                elif cat == "rfs":
                    has = [s.strip() for s in tags.get("has", "").split(",")]
                    rfs_types.append(RFSDef(
                        tags["name"], header, parse_params_str(tags.get("params", "")),
                        init_methods=[s.strip() for s in tags.get("init", "").split(",") if s.strip()],
                        has_cardinality="cardinality" in has,
                        has_track_histories="track_histories" in has,
                        has_birth_weights="birth_weights" in has,
                        has_cluster_object="cluster_object" in has,
                        optional_params=parse_args_str(tags.get("optional_params", ""))))
                elif cat == "clustering":
                    clustering.append(ClusteringDef(tags["name"], header,
                                                    parse_args_str(tags.get("args", ""))))
                elif cat == "icp":
                    icp_types.append(IcpDef(tags["name"], header,
                                            tags.get("namespace", "template_matching"),
                                            parse_args_str(tags.get("args", "")),
                                            parse_params_str(tags.get("params", ""))))
    return dynamics, models_list, filters, rfs_types, clustering, icp_types


# ---------------------------------------------------------------------------
# pybind11 emitters
# ---------------------------------------------------------------------------
CPP_ARG = {"int": "int", "scalar": "double", "double": "double",
           "vec": "Eigen::VectorXd", "mat": "Eigen::MatrixXd",
           "ivec": "std::vector<int>", "fvec": "std::vector<double>"}

def _ctor_type(t):
    # clustering 'vec'/'ivec' map to std::vector (raw lists), everything else per CPP_ARG
    return CPP_ARG.get(t, "double")

def emit_includes(defs_headers):
    L = ["// Auto-generated from @mex annotations by brewpy/generate_pybind.py. DO NOT EDIT.",
         "#include <pybind11/pybind11.h>", "#include <pybind11/stl.h>",
         "#include <pybind11/eigen.h>", "#include <Eigen/Dense>",
         "#include <memory>", "#include <stdexcept>", "#include <string>",
         "#include <utility>", "#include <vector>", ""]
    for h in sorted(defs_headers):
        L.append(f'#include "{h}"')
    L += ["", "namespace py = pybind11;", "using namespace brew;", ""]
    return "\n".join(L)

def emit_dynamics(dynamics):
    L = ['  auto dyn = m.def_submodule("dynamics");',
         '  py::class_<dynamics::DynamicsBase<>, std::shared_ptr<dynamics::DynamicsBase<>>>(dyn, "DynamicsBase");']
    for d in dynamics:
        impl = f"dynamics::{d.name}<>"
        bound = f"BrewpyDynamicsAdapter<{impl}>" if d.fixed_state_dim else impl
        types = ", ".join(_ctor_type(t) for _, t in d.args)
        args = "".join(f', py::arg("{n}")' for n, _ in d.args)
        init = f".def(py::init<{types}>(){args})" if d.args else ".def(py::init<>())"
        L.append(f'  py::class_<{bound}, dynamics::DynamicsBase<>, '
                 f'std::shared_ptr<{bound}>>(dyn, "{d.name}")\n      {init}')
        # Nonlinear dynamics need the evaluation state for their Jacobian and input map.
        # Keep the dt-only convenience overload, but make it deterministic by evaluating at
        # the zero state instead of relying on Eigen's empty or uninitialised default vector.
        for py_name, cpp_name in (("state_matrix", "get_state_mat"),
                                  ("input_matrix", "get_input_mat")):
            L.append(f'      .def("{py_name}", [](const {bound}& d, double dt, '
                     f'const Eigen::VectorXd& x) {{ return d.{cpp_name}(dt, x); }}, '
                     f'py::arg("dt"), py::arg("state"))')
            L.append(f'      .def("{py_name}", [](const {bound}& d, double dt) {{ '
                     f'const auto names = d.state_names(); '
                     f'return d.{cpp_name}(dt, Eigen::VectorXd::Zero(names.size())); }}, '
                     f'py::arg("dt"))')
        L.append(f'      .def("propagate", [](const {bound}& d, double dt, '
                 f'const Eigen::VectorXd& x) {{ return d.propagate_state(dt, x); }}, '
                 f'py::arg("dt"), py::arg("state"))')
        if d.name == "ClohessyWiltshire":
            L.append(f'      .def("acceleration", &dynamics::{d.name}<>::acceleration, '
                     f'py::arg("state"));')
        else:
            L[-1] += ";"
    return "\n".join(L)

def emit_clustering(clustering):
    L = ['  auto clt = m.def_submodule("clustering");',
         '  py::class_<clustering::ClusterBase, std::shared_ptr<clustering::ClusterBase>>(clt, "ClusterBase");']
    for c in clustering:
        if c.name == "MahalanobisDBSCAN":
            continue  # derives from DBSCAN, not ClusterBase -> invalid class_<.,ClusterBase,.>
        tmap = {"vec": "std::vector<double>", "ivec": "std::vector<int>",
                "int": "int", "scalar": "double", "double": "double"}
        types = ", ".join(tmap.get(t, "double") for _, t in c.args)
        args = "".join(f', py::arg("{n}")' for n, _ in c.args)
        L.append(f'  py::class_<clustering::{c.name}, clustering::ClusterBase, '
                 f'std::shared_ptr<clustering::{c.name}>>(clt, "{c.name}")')
        L.append(f'      .def(py::init<{types}>(){args})')
        L.append(f'      .def("cluster", &clustering::{c.name}::cluster, py::arg("Z"));')
    return "\n".join(L)

def emit_filters(filters):
    L = ['  auto flt = m.def_submodule("filters");']
    for f in filters:
        T = f"filters::{f.name}<>"
        L.append(f'  py::class_<{T}>(flt, "{f.name}")')
        L.append("      .def(py::init<>())")
        L.append(f'      .def("set_dynamics", &{T}::set_dynamics, py::arg("dyn"))')
        for meth, arg in (("set_process_noise", "Q"), ("set_measurement_jacobian", "H"),
                          ("set_measurement_noise", "R")):
            if f.name == "USGGIWEKF" and meth == "set_measurement_jacobian":
                continue
            if meth == "set_measurement_jacobian":
                L.append(f'      .def("{meth}", []({T}& f, const Eigen::MatrixXd& {arg}) '
                         f'{{ typename {T}::MeasMatrix fixed = {arg}; f.{meth}(std::move(fixed)); }}, '
                         f'py::arg("{arg}"))')
            else:
                L.append(f'      .def("{meth}", []({T}& f, const Eigen::MatrixXd& {arg}) '
                         f'{{ f.{meth}({arg}); }}, py::arg("{arg}"))')
        for sname, stype in f.setters:
            if stype in ("mat", "vec"):
                ct = CPP_ARG[stype]
                L.append(f'      .def("set_{sname}", []({T}& f, const {ct}& v) '
                         f'{{ f.set_{sname}(v); }}, py::arg("{sname}"))')
            else:
                L.append(f'      .def("set_{sname}", &{T}::set_{sname}, py::arg("{sname}"))')
        for sname, _ in f.handle_setters:
            L.append(f'      .def("set_{sname}", &{T}::set_{sname}, py::arg("{sname}"))')
        L[-1] += ";"
    return "\n".join(L)

def _accessor(cpp_type, field):
    n, t = field.name, field.type
    if t in ("vec",):
        return (f'.def_property_readonly("{n}", []({cpp_type}& c) '
                f'{{ return Eigen::VectorXd(c.{n}()); }})')
    if t in ("mat",):
        return (f'.def_property_readonly("{n}", []({cpp_type}& c) '
                f'{{ return Eigen::MatrixXd(c.{n}()); }})')
    # scalar: wrap in a const-lambda so the const overload is selected unambiguously
    # (models expose both `T name() const` and `T& name()`, making &Type::name ambiguous).
    return (f'.def_property_readonly("{n}", [](const {cpp_type}& c) '
            f'{{ return c.{n}(); }})')

def emit_models(models_list):
    L = ['  auto mdl = m.def_submodule("models");']
    for mo in models_list:
        if mo.is_trajectory:
            continue                                    # trajectory models: handled in static prelude
        T = mo.cpp_type
        if mo.name == "TemplatePose":
            types = "Eigen::VectorXd, Eigen::MatrixXd, int, std::vector<int>, Eigen::MatrixXd, int"
            args = ', py::arg("mean"), py::arg("covariance"), py::arg("template_id"), py::arg("pos_indices"), py::arg("rotation") = Eigen::Matrix3d::Identity(), py::arg("mode_id") = 0'
            init = (f'.def(py::init([](Eigen::VectorXd mean, Eigen::MatrixXd covariance, int template_id, '
                    f'std::vector<int> pos_indices, Eigen::MatrixXd rotation, int mode_id) {{ return {T}(std::move(mean), '
                    f'std::move(covariance), template_id, std::move(pos_indices), rotation, mode_id); }}){args})')
        else:
            types = ", ".join(CPP_ARG.get(f.type, "double") for f in mo.fields)
            args = "".join(f', py::arg("{f.name}")' for f in mo.fields)
            init = f'.def(py::init<{types}>(){args})'
        L.append(f'  py::class_<{T}>(mdl, "{mo.name}")')
        L.append(f'      {init}')
        accs = [_accessor(T, f) for f in mo.fields + mo.extract_extra]
        for a in accs:
            L.append(f"      {a}")
        for n in mo.create_int_fields:
            L.append(f'      .def_property_readonly("{n}", [](const {T}& c) {{ return c.{n}(); }})')
        for n in mo.create_int_vec_fields:
            L.append(f'      .def_property_readonly("{n}", [](const {T}& c) {{ return c.{n}(); }})')
        if mo.name == "TemplatePose":
            L.append(f'      .def_property_readonly("mode_id", [](const {T}& c) {{ return c.mode_id(); }})')
            L.append(f'      .def("mark_pca_aligned", &{T}::mark_pca_aligned)')
            for name in ("impulse_observation", "impulse_variance", "impulse_branch",
                         "impulse_log_zero", "impulse_log_positive"):
                L.append(f'      .def_property_readonly("{name}", &{T}::{name})')
        L[-1] += ";"
        # Mixture<Model>
        MT = f"models::Mixture<{T}>"
        L.append(f'  py::class_<{MT}>(mdl, "{mo.name}Mixture")')
        L.append("      .def(py::init<>())")
        L.append(f'      .def("add_component", []({MT}& mx, const {T}& c, double w) '
                 f'{{ mx.add_component({T}(c), w); }}, py::arg("component"), py::arg("weight"))')
        L.append(f'      .def("size", &{MT}::size).def("__len__", &{MT}::size)')
        L.append(f'      .def("weight", &{MT}::weight, py::arg("i"))')
        L.append(f'      .def("component", []({MT}& mx, std::size_t i) -> {T}& '
                 f'{{ return mx.component(i); }}, py::arg("i"), py::return_value_policy::reference_internal);')
    return "\n".join(L)

def emit_rfs(models_list, rfs_types, filters):
    """Bind each concrete (model-package)::RFS<> tracker, one submodule per package."""
    L = []
    dist2filters = {}
    for f in filters:
        if f.dist:
            dist2filters.setdefault(f.dist, []).append(f.name)
    rfs_models = [m for m in models_list if m.name not in SKIP_RFS_MODELS and not m.is_trajectory]
    pkgs = {}
    for mo in rfs_models:
        pkgs.setdefault(pkg_of(mo), mo)               # one representative model per package
    for pkg, mo in pkgs.items():
        T = mo.cpp_type
        MT = f"models::Mixture<{T}>"
        FTS = dist2filters.get(mo.name, [])
        if not FTS:
            continue                                  # no matching filter -> tracker bound elsewhere
        sub = f"{pkg}_mod"
        L.append(f'  auto {sub} = m.def_submodule("{pkg}");')
        for r in (item for item in rfs_types if pkg_of(item) == pkg):
            if r.name in ("MBM", "PMBM"):
                continue  # different (Bernoulli) birth API; not needed for the constellation sim
            RT = f"{pkg}::{r.name}<>"
            L.append(f'  py::class_<{RT}>({sub}, "{r.name}")')
            L.append("      .def(py::init<>())")
            for FT in FTS:
                L.append(f'      .def("set_filter", []({RT}& p, const filters::{FT}<>& f) '
                         f'{{ p.set_filter(f.clone()); }}, py::arg("filter"))')
            if "GLMB" in r.name:
                L.append(f'      .def("set_birth_model", []({RT}& p, const {MT}& b, double pb) '
                         f'{{ p.set_birth_model(b.clone(), pb); }}, py::arg("birth"), py::arg("p_birth"))')
                # measurement-driven multi-target birth: one birth term (=potential labelled track)
                # per detection, each with its own birth probability. set_birth_model only takes a
                # single term (one multimodal track), which cannot birth several targets at once.
                L.append(f'      .def("set_birth_terms", []({RT}& p, const std::vector<{MT}*>& mixes, '
                         f'const std::vector<double>& pbs) {{ '
                         f'std::vector<std::pair<std::unique_ptr<{MT}>, double>> terms; '
                         f'for (std::size_t i = 0; i < mixes.size(); ++i) '
                         f'terms.emplace_back(mixes[i]->clone(), i < pbs.size() ? pbs[i] : 0.1); '
                         f'p.set_birth_terms(std::move(terms)); }}, py::arg("mixtures"), py::arg("p_births"))')
            else:
                L.append(f'      .def("set_birth_model", []({RT}& p, const {MT}& b) '
                         f'{{ p.set_birth_model(b.clone()); }}, py::arg("birth"))')
            if r.name in ("PHD", "CPHD"):
                # the (P)PHD intensity is null until set; predict() no-ops without it
                L.append(f'      .def("set_intensity", []({RT}& p, const {MT}& i) '
                         f'{{ p.set_intensity(i.clone()); }}, py::arg("intensity"))')
            if (pkg in ("ggiw", "usggiw") and r.name == "PHD") or \
                    (pkg in ("ggiw", "usggiw") and r.name == "GLMB") or \
                    (pkg == "gaussian" and r.name == "GLMB"):
                # state-dependent detection probability (observation-window / field-of-view aware):
                # ggiw PHD onboard (per-satellite FOV) and the ground GLMB (observability field from the
                # known satellite sub-points, so out-of-view storms coast).
                L.append(f'      .def("set_detection_field", []({RT}& p, const Eigen::MatrixXd& field, '
                         f'double radius, double pd_out) {{ p.set_detection_field(field, radius, pd_out); }}, '
                         f'py::arg("field"), py::arg("radius"), py::arg("pd_out"))')
            if pkg in ("ggiw", "usggiw") and r.name == "GLMB":
                # ingest the fused PHD mixture as GGIW measurement components (extent + rate carried),
                # fused via the GGIW measurement update (correct_component)
                L.append(f'      .def("set_measurement_components", []({RT}& p, const {MT}& z) '
                         f'{{ p.set_measurement_components(z); }}, py::arg("z"))')
            if pkg == "usggiw" and r.name == "PHD":
                L.append(f'      .def("set_measurement_components", []({RT}& p, const {MT}& z) '
                         f'{{ p.set_measurement_components(z); }}, py::arg("z"))')
            for meth in r.init_methods:
                pass  # @mex_init is MEX-store specific
            for setter, default in (("set_prob_detection", "pd"), ("set_prob_survive", "ps"),
                                    ("set_clutter_rate", "rate"), ("set_clutter_density", "density")):
                L.append(f'      .def("{setter}", &{RT}::{setter}, py::arg("{default}"))')
            for pname, _, _ in r.params:
                L.append(f'      .def("set_{pname}", &{RT}::set_{pname}, py::arg("{pname}"))')
            for opname, _ in r.optional_params:
                L.append(f'      .def("set_{opname}", &{RT}::set_{opname}, py::arg("{opname}"))')
            if r.has_cluster_object:
                L.append(f'      .def("set_cluster_object", &{RT}::set_cluster_object, py::arg("obj"))')
            L.append(f'      .def("predict", &{RT}::predict, py::arg("timestep"), py::arg("dt"))')
            L.append(f'      .def("correct", &{RT}::correct, py::arg("measurements"))')
            L.append(f'      .def("cleanup", &{RT}::cleanup)')
            if r.has_cardinality:
                L.append(f'      .def("estimated_cardinality", &{RT}::estimated_cardinality)')
            if r.has_track_histories:
                L.append(f'      .def("track_histories", &{RT}::track_histories)')
            L.append(f'      .def("latest", []({RT}& p) {{ return brewpy_latest<{RT}, {MT}>(p); }});')
    return "\n".join(L)


def generate(brew_root: Path) -> str:
    inc = brew_root / "include"
    dynamics, models_list, filters, rfs_types, clustering, icp_types = scan_headers(inc)
    headers = set()
    for grp in (dynamics, models_list, filters, rfs_types, clustering, icp_types):
        for d in grp:
            headers.add(d.header)
    for mo in models_list:
        headers.add(f"brew/{pkg_of(mo)}/{pkg_of(mo)}.hpp")
    headers.update(["brew/shared/mixture.hpp", "brew/dynamics/dynamics_base.hpp",
                    "brew/clustering/cluster_base.hpp", "brew/shared/filter_base.hpp"])

    static = (Path(__file__).parent / "pybind_static.cpp.in").read_text(encoding="utf-8")
    body = [emit_includes(headers), static,
            "PYBIND11_MODULE(_brewpy, m) {",
            '  m.doc() = "BREW RFS tracking (pybind11), auto-generated from @mex annotations.";',
            emit_models(models_list), emit_dynamics(dynamics), emit_clustering(clustering),
            emit_filters(filters), emit_rfs(models_list, rfs_types, filters),
            "  brewpy_static_extra(m);", "}", ""]
    counts = (f"// generated: {len(dynamics)} dynamics, {len(models_list)} models, "
              f"{len(filters)} filters, {len(rfs_types)} rfs, {len(clustering)} clustering, "
              f"{len(icp_types)} icp")
    return counts + "\n" + "\n".join(body)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brew", default=str(Path(__file__).resolve().parent.parent / "brew"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    brew_root = Path(a.brew).resolve()
    code = generate(brew_root)
    out = Path(a.out) if a.out else (brew_root / "bindings" / "bindings_generated.cpp")
    out.write_text(code, encoding="utf-8")
    print(f"[gen] wrote {out}")
    print(code.splitlines()[0])


if __name__ == "__main__":
    _main()
