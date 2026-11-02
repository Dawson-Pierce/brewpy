#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "brew/models/merge/usggiw.hpp"
#include "brew/models/merge/trajectory_usggiw.hpp"
#include "brew/filters/usggiw_ekf.hpp"
#include "brew/filters/trajectory_usggiw_ekf.hpp"
#include "brew/multiObjectTrackers/phd.hpp"
#include "brew/utils/fusion/usggiw_gci.hpp"

namespace py = pybind11;
using namespace brew;

void brewpy_bind_spherical_tracking(py::module_& m) {
    auto sphere = m.def_submodule("multiObjectTrackers");
    auto trajectory = m.def_submodule("multiObjectTrackers");
    using Config = multiObjectTrackers::SphericalPHDConfig;
    using OnboardPHD = brew::multiObjectTrackers::PHD<models::USGGIW<double, Eigen::Dynamic>, Eigen::Dynamic>;
    using GroundPHD = brew::multiObjectTrackers::PHD<models::TrajectoryUSGGIW<double, Eigen::Dynamic>, Eigen::Dynamic>;
    using Fusion = utils::fusion::ReportFusion;
    using Mix = utils::fusion::SphericalMixture;
    py::class_<Config>(sphere, "SphericalPHDConfig")
        .def(py::init([](py::kwargs kwargs) {
            Config config;
            for (auto item : kwargs) {
                const auto name = py::cast<std::string>(item.first);
                if (name == "process_angle_var") config.process_angle_var = py::cast<double>(item.second);
                else if (name == "process_rate_var") config.process_rate_var = py::cast<double>(item.second);
                else if (name == "measurement_sigma_deg") config.measurement_sigma_deg = py::cast<double>(item.second);
                else if (name == "temporal_decay") config.temporal_decay = py::cast<double>(item.second);
                else if (name == "forgetting_factor") config.forgetting_factor = py::cast<double>(item.second);
                else if (name == "scaling_parameter") config.scaling_parameter = py::cast<double>(item.second);
                else if (name == "ci_weight") config.ci_weight = py::cast<double>(item.second);
                else if (name == "prob_detection") config.prob_detection = py::cast<double>(item.second);
                else if (name == "prob_survive") config.prob_survive = py::cast<double>(item.second);
                else if (name == "clutter_rate") config.clutter_rate = py::cast<double>(item.second);
                else if (name == "clutter_density") config.clutter_density = py::cast<double>(item.second);
                else if (name == "extract_threshold") config.extract_threshold = py::cast<double>(item.second);
                else if (name == "prune_threshold") config.prune_threshold = py::cast<double>(item.second);
                else if (name == "merge_threshold") config.merge_threshold = py::cast<double>(item.second);
                else if (name == "gate_threshold") config.gate_threshold = py::cast<double>(item.second);
                else if (name == "max_components") config.max_components = py::cast<int>(item.second);
                else if (name == "cluster_eps_deg") config.cluster_eps_deg = py::cast<double>(item.second);
                else if (name == "cluster_min_pts") config.cluster_min_pts = py::cast<int>(item.second);
                else if (name == "birth_weight") config.birth_weight = py::cast<double>(item.second);
                else if (name == "birth_alpha") config.birth_alpha = py::cast<double>(item.second);
                else if (name == "birth_beta") config.birth_beta = py::cast<double>(item.second);
                else if (name == "birth_angle_sigma_deg") config.birth_angle_sigma_deg = py::cast<double>(item.second);
                else if (name == "birth_rate_sigma_deg") config.birth_rate_sigma_deg = py::cast<double>(item.second);
                else if (name == "birth_extent_sigma_deg") config.birth_extent_sigma_deg = py::cast<double>(item.second);
                else if (name == "birth_extent_dof") config.birth_extent_dof = py::cast<double>(item.second);
                else if (name == "birth_suppress_deg") config.birth_suppress_deg = py::cast<double>(item.second);
                else if (name == "fov_radius_deg") config.fov_radius_deg = py::cast<double>(item.second);
                else if (name == "prob_detection_out") config.prob_detection_out = py::cast<double>(item.second);
                else throw py::type_error("Unknown tracker setting: " + name);
            }
            return config;
        }))
        .def("__copy__", [](const Config& c) { return c; })
        .def("__deepcopy__", [](const Config& c, py::dict) { return c; })
        .def_readwrite("process_angle_var", &Config::process_angle_var)
        .def_readwrite("process_rate_var", &Config::process_rate_var)
        .def_readwrite("measurement_sigma_deg", &Config::measurement_sigma_deg)
        .def_readwrite("temporal_decay", &Config::temporal_decay)
        .def_readwrite("forgetting_factor", &Config::forgetting_factor)
        .def_readwrite("scaling_parameter", &Config::scaling_parameter)
        .def_readwrite("ci_weight", &Config::ci_weight)
        .def_readwrite("prob_detection", &Config::prob_detection)
        .def_readwrite("prob_survive", &Config::prob_survive)
        .def_readwrite("clutter_rate", &Config::clutter_rate)
        .def_readwrite("clutter_density", &Config::clutter_density)
        .def_readwrite("extract_threshold", &Config::extract_threshold)
        .def_readwrite("prune_threshold", &Config::prune_threshold)
        .def_readwrite("merge_threshold", &Config::merge_threshold)
        .def_readwrite("gate_threshold", &Config::gate_threshold)
        .def_readwrite("max_components", &Config::max_components)
        .def_readwrite("cluster_eps_deg", &Config::cluster_eps_deg)
        .def_readwrite("cluster_min_pts", &Config::cluster_min_pts)
        .def_readwrite("birth_weight", &Config::birth_weight)
        .def_readwrite("birth_alpha", &Config::birth_alpha)
        .def_readwrite("birth_beta", &Config::birth_beta)
        .def_readwrite("birth_angle_sigma_deg", &Config::birth_angle_sigma_deg)
        .def_readwrite("birth_rate_sigma_deg", &Config::birth_rate_sigma_deg)
        .def_readwrite("birth_extent_sigma_deg", &Config::birth_extent_sigma_deg)
        .def_readwrite("birth_extent_dof", &Config::birth_extent_dof)
        .def_readwrite("birth_suppress_deg", &Config::birth_suppress_deg)
        .def_readwrite("fov_radius_deg", &Config::fov_radius_deg)
        .def_readwrite("prob_detection_out", &Config::prob_detection_out);

    py::reinterpret_borrow<py::class_<OnboardPHD>>(sphere.attr("USGGIWPHD"))
        .def(py::init([](const Config& config) {
            filters::USGGIWEKF<> filter;
            auto phd = std::make_unique<OnboardPHD>();
            config.configure(filter, *phd, utils::sphere::angular_variance(config.measurement_sigma_deg));
            filter.set_application_config(config);
            phd->set_filter(filter.clone());
            phd->set_intensity(std::make_unique<Mix>());
            phd->set_birth_model(std::make_unique<Mix>());
            phd->set_cluster_object(std::make_shared<utils::clustering::DBSCAN>(
                2.*std::sin(.5*utils::sphere::radians(config.cluster_eps_deg)), config.cluster_min_pts,
                std::vector<int>{0,1,2}, std::vector<double>{1.,1.,1.}));
            phd->set_detection_field(Eigen::MatrixXd(3,0), utils::sphere::radians(config.fov_radius_deg), config.prob_detection_out);
            return phd;
        }), py::arg("config"))
        .def_property_readonly("config", [](const OnboardPHD& phd) {
            return dynamic_cast<const filters::USGGIWEKF<>&>(phd.filter()).application_config();
        })
        .def("step", [](OnboardPHD& phd, const Eigen::MatrixXd& measurements, double dt, int frame,
                        const std::optional<Eigen::MatrixXd>& observers) {
            if (observers && phd.detection_radius() > 0.)
                phd.set_detection_field(utils::sphere::lonlat_directions(*observers), phd.detection_radius(),
                    phd.outside_detection_probability());
            return phd.step(measurements, dt, frame);
        }, py::arg("measurements_lonlat"), py::arg("dt_min"),
             py::arg("frame") = 0, py::arg("observer_lonlat") = py::none())
        .def("estimates", &OnboardPHD::estimates)
        .def("mixture", &OnboardPHD::mixture, py::arg("weight_min") = py::none())
        .def("report_mixture", [](const OnboardPHD& phd, const Eigen::MatrixXd& observer_lonlat) {
            auto result = std::make_unique<Mix>();
            const auto selected = phd.mixture();
            const auto centers = utils::sphere::lonlat_directions(observer_lonlat);
            if (centers.cols()) for (std::size_t i = 0; i < selected->size(); ++i)
                if (phd.filter().detection_probability(selected->component(i), centers,
                        phd.detection_radius(), 1., 0.) > 0.)
                    result->add_component(selected->component(i), selected->weight(i));
            return result;
        }, py::arg("observer_lonlat"));
    py::reinterpret_borrow<py::class_<GroundPHD>>(trajectory.attr("TrajectoryUSGGIWPHD"))
        .def(py::init([](const Config& config, double sigma, double birth_position, double birth_rate,
                         int window, double dedup, double velocity_sigma) {
            filters::TrajectoryUSGGIWEKF<> filter;
            auto phd = std::make_unique<GroundPHD>();
            config.configure(filter, *phd, utils::sphere::angular_variance(sigma));
            filter.set_application_config(config, sigma, birth_position, birth_rate, window, dedup, velocity_sigma);
            phd->set_filter(filter.clone());
            phd->set_intensity(std::make_unique<GroundPHD::Mixture>());
            phd->set_birth_model(std::make_unique<GroundPHD::Mixture>());
            phd->set_detection_field(Eigen::MatrixXd(3,0), utils::sphere::radians(config.fov_radius_deg), config.prob_detection_out);
            return phd;
        }),
             py::arg("config"), py::arg("measurement_sigma_deg"), py::arg("birth_pos_var_deg2"),
             py::arg("birth_rate_var_deg2_min2"), py::arg("window_size") = 8,
             py::arg("dedup_deg") = 0., py::arg("velocity_sigma_deg_min") = 0.)
        .def_property_readonly("config", [](const GroundPHD& phd) {
            return dynamic_cast<const filters::TrajectoryUSGGIWEKF<>&>(phd.filter()).application_config();
        })
        .def_property_readonly("current_extracted_histories", [](const GroundPHD& tracker) {
            // Python may retain these arrays across updates that replace native storage.
            return tracker.current_extracted_histories();
        })
        .def("step", [](GroundPHD& phd, const Mix& reports, double dt, const Eigen::MatrixXd& observers,
                        const std::optional<std::vector<int>>& groups) {
            phd.set_measurement_components(reports);
            phd.set_grouped_detection_field(utils::sphere::lonlat_directions(observers), groups.value_or(std::vector<int>{}),
                phd.detection_radius(), phd.outside_detection_probability());
            return phd.step(Eigen::MatrixXd(3,0), dt);
        }, py::arg("mixture"), py::arg("dt_min"),
             py::arg("observer_lonlat"), py::arg("observer_groups") = py::none())
        .def("track_histories", &GroundPHD::track_histories)
        .def("estimates", &GroundPHD::estimates)
        .def("extracted_cardinality", &GroundPHD::extracted_cardinality);
    auto fusion = m.def_submodule("utils").def_submodule("fusion");
    py::class_<Fusion>(fusion, "ReportFusion")
        .def(py::init<double>(), py::arg("gate_deg") = 5.)
        .def("reduce", &Fusion::reduce, py::arg("mixture"))
        .def_static("unite", [](py::args inputs) {
            std::vector<const Mix*> mixtures;
            for (auto input : inputs) mixtures.push_back(py::cast<const Mix*>(input));
            return Fusion::unite(mixtures);
        })
        .def("fuse_rings", [](const Fusion& fusion,
                             const std::vector<std::vector<const Mix*>>& reports) {
            auto result = fusion.fuse_rings(reports);
            py::list rings;
            for (auto& ring : result.rings) rings.append(py::cast(std::move(ring)));
            return py::make_tuple(rings, py::cast(std::move(result.pooled)));
        }, py::arg("ordered_reports"));
    // Geometry helpers are also used to visualize velocities and compare truth.
    auto geometry = m.def_submodule("utils").def_submodule("sphere");
    geometry.def("spatial_basis", &utils::sphere::spatial_basis);
    geometry.def("transport_matrix", &utils::sphere::transport_matrix);
    geometry.def("propagate", [](const Eigen::Vector3d& r, const Eigen::Vector2d& v, double dt) {
        auto next = utils::sphere::propagate(r, v, dt);
        return py::make_tuple(next.direction, next.velocity);
    });
}
