"""Check annotation handling without importing the compiled extension."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "generate_pybind", ROOT / "generator_files/generate_pybind.py")
generator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = generator
spec.loader.exec_module(generator)


class GeneratorTests(unittest.TestCase):
    def test_public_headers_use_role_folders(self):
        include = ROOT / "brew/include/brew"
        self.assertEqual({p.name for p in include.iterdir() if p.is_dir()},
                         {"models", "dynamics", "filters", "multiObjectTrackers", "utils"})

    def test_generation_is_deterministic(self):
        self.assertEqual(generator.generate(ROOT / "brew"),
                         generator.generate(ROOT / "brew"))

    def test_unknown_argument_type_is_rejected(self):
        definition = generator.DynamicsDef("Example", "example.hpp", [("value", "unknown")])
        with self.assertRaisesRegex(KeyError, "unknown"):
            generator.emit_dynamics([definition])

    def test_configuration_uses_native_field_types_and_rejects_unknown_names(self):
        _, _, filters, rfs, _, _ = generator.scan_headers(ROOT / "brew/include")
        emitted = generator.emit_config_parsers([*filters, *rfs])
        self.assertIn('py::cast<decltype(config.process_noise)>(item.second)', emitted)
        self.assertIn('py::cast<decltype(config.birth_model)>(item.second)', emitted)
        self.assertIn('Unknown TmEkf configuration field:', emitted)
        self.assertIn('Unknown PHD configuration field:', emitted)
        self.assertNotIn('0.99', emitted)

    def test_intent_bindings_follow_density_metadata(self):
        emitted = generator.generate(ROOT / "brew")
        for density, package in [('Gaussian', 'gaussian'), ('GGIW', 'ggiw'), ('USGGIW', 'usggiw')]:
            self.assertIn(f'brewpy_bind_categorical_intent<models::{density}<>>(m, "{density}")', emitted)
        self.assertIn('const filters::Filter<TG>&', emitted)
        self.assertIn('const filters::Filter<TUS>&', emitted)

    def test_rfs_constructors_only_accept_matching_filters(self):
        _, models, filters, rfs, _, _ = generator.scan_headers(ROOT / "brew/include")
        emitted = generator.emit_rfs(models, rfs, filters)
        start = emitted.index('py::class_<multiObjectTrackers::PHD<models::TemplatePose<>>>')
        end = emitted.index('py::class_<', start + 1)
        binding = emitted[start:end]
        self.assertIn('const filters::Filter<models::TemplatePose<>>& f', binding)
        self.assertNotIn('Filter<models::GGIW<>>', binding)
        self.assertNotIn('set_intent_estimator', binding)
        emitted_filters = generator.emit_filters(filters)
        self.assertIn('py::class_<filters::TmEkf<>, filters::Filter<models::TemplatePose<>>>', emitted_filters)
        self.assertIn('py::class_<filters::ImmTmEkf<>, filters::Filter<models::TemplatePose<>>>', emitted_filters)
        self.assertNotIn('set_intent_estimator', emitted_filters)


if __name__ == "__main__":
    unittest.main()
