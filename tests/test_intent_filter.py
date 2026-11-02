"""The intent wrapper is a density-compatible RFS filter with independent copies."""
import gc
import unittest
import numpy as np
import brewpy

tp = brewpy.models


def intent():
    return brewpy.filters.IntentEstimator([
        tp.IntentHypothesis(str(p), tp.PointingFiringModel(
            tp.PointingRule.Fixed, firing_probability=p, impulse_max=.2))
        for p in [0., .01]
    ])


def density(x):
    return brewpy.models.TemplatePose(np.array([x, 0., 0., 0., 0., 0.]), np.eye(9), 0, [0, 1, 2])


def mixture(pose):
    result = brewpy.models.TemplatePoseMixture()
    result.add_component(pose, 1.)
    return result


class IntentFilterTests(unittest.TestCase):
    def test_matching_rfs_construction_and_type_rejection(self):
        for primary in [brewpy.filters.TmEkf(), brewpy.filters.ImmTmEkf()]:
            wrapper = brewpy.filters.IntentFilter(primary, intent())
            self.assertIsInstance(wrapper, brewpy.filters.TemplatePoseFilter)
            self.assertFalse(hasattr(primary, 'set_intent_estimator'))
            estimator = brewpy.multiObjectTrackers.PHD(wrapper)
            estimator.set_filter(wrapper)
            with self.assertRaises(TypeError):
                brewpy.multiObjectTrackers.GGIWPHD(wrapper)
        with self.assertRaises(TypeError):
            brewpy.filters.IntentFilter(brewpy.filters.GGIWEKF(), intent())

    def test_rfs_owns_independent_wrapper_copies(self):
        primary = brewpy.filters.TmEkf()
        points = np.random.default_rng(31).normal(size=(3, 36))
        library = brewpy.utils.registration.TemplateLibrary()
        library.add(points)
        primary.set_template_library(library)
        primary.set_icp(brewpy.utils.registration.PointToPointIcp())
        motion = brewpy.dynamics.ClohessyWiltshire(.001)
        motion.set_impulse_model(brewpy.dynamics.ImpulseModel(tp.ImpulsePrior(.2, .01), body_frame=True))
        primary.set_dynamics(motion)
        primary.set_process_noise(np.eye(6) * .001)
        primary.set_measurement_noise(np.eye(6) * .01)
        primary.set_rotation_process_noise(np.eye(3) * .001)
        primary.set_measurement_jacobian(np.hstack([np.eye(3), np.zeros((3, 3))]))
        hmm = intent()
        wrapper = brewpy.filters.IntentFilter(primary, hmm)
        a, b, plain = brewpy.multiObjectTrackers.PHD(wrapper), brewpy.multiObjectTrackers.PHD(wrapper), brewpy.multiObjectTrackers.PHD(primary)
        del primary, hmm, wrapper
        gc.collect()
        initial = density(100.)
        initial.mark_pca_aligned()
        mark = brewpy.models.CategoricalIntent()
        mark.objectives = ['0.0', '0.01']; mark.probabilities = [.8, .2]
        initial.intent = mark
        for estimator in [a, b, plain]:
            estimator.set_intensity(mixture(initial))
            estimator.set_prob_detection(.8)
            estimator.set_gate_threshold(1e6)
            estimator.set_extract_threshold(0.)
            estimator.predict(1, 1.)
        measurements = points + np.array([[100.1], [0.], [0.]])
        a.correct(measurements)
        plain.correct(measurements)
        a.cleanup()
        plain.cleanup()
        self.assertEqual(a.latest().size(), plain.latest().size())
        for i in range(a.latest().size()):
            self.assertEqual(a.latest().weight(i), plain.latest().weight(i))
            np.testing.assert_array_equal(a.latest().component(i).mean, plain.latest().component(i).mean)
            np.testing.assert_array_equal(a.latest().component(i).covariance, plain.latest().component(i).covariance)
            for field in ['probability', 'magnitude', 'std', 'lower', 'upper']:
                actual = getattr(a.latest().component(i).firing, 'impulse_' + field)
                self.assertTrue(np.isfinite(actual))
                self.assertEqual(actual, getattr(plain.latest().component(i).firing, 'impulse_' + field))
        before = a.latest().component(0).intent.probabilities.copy()
        b.correct(np.empty((3, 0)))
        b.cleanup()
        np.testing.assert_allclose(b.latest().component(0).intent.probabilities, [.74, .26], atol=1e-14)
        np.testing.assert_array_equal(a.latest().component(0).intent.probabilities, before)
        b.cleanup()
        np.testing.assert_allclose(b.latest().component(0).intent.probabilities, [.74, .26], atol=1e-14)


if __name__ == '__main__':
    unittest.main()
