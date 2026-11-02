"""Categorical attachments use the same filter/mixture interfaces across densities."""
import unittest
import numpy as np
import brewpy


def mark(probabilities, names=('a', 'b')):
    result = brewpy.models.CategoricalIntent()
    result.objectives = list(names)
    result.probabilities = probabilities
    return result


def estimator(density):
    model = getattr(brewpy.models, density + 'UninformativeBehavior')
    return brewpy.filters.IntentEstimator([
        brewpy.models.IntentHypothesis(name, model()) for name in ('a', 'b')])


def gaussian(x):
    return brewpy.models.Gaussian(np.array([x, 0., .1, 0.]), np.eye(4))


class CategoricalIntentTests(unittest.TestCase):
    def test_public_api_uses_roles_and_native_rfs_specializations(self):
        for name in ('models', 'dynamics', 'filters', 'multiObjectTrackers', 'utils'):
            self.assertTrue(hasattr(brewpy, name), name)
        for name in ('template_pose', 'gaussian', 'ggiw', 'usggiw', 'intent'):
            self.assertFalse(hasattr(brewpy, name), name)
        native = brewpy.multiObjectTrackers.PHD(brewpy.filters.EKF())
        self.assertIsInstance(native, brewpy.multiObjectTrackers.GaussianPHD)

    def test_generic_hmm_analyzes_known_density_histories(self):
        hmm = estimator("Gaussian")
        history = [gaussian(0.), gaussian(1.), gaussian(2.)]
        np.testing.assert_array_equal(hmm.posterior_history(history), np.full((3, 2), .5))
        for density in history:
            self.assertEqual(len(density.intent.probabilities), 0)
        hmm.correct(history[0])
        hmm.correct(history[1], history[0])
        np.testing.assert_array_equal(history[1].intent.probabilities, [.5, .5])
        np.testing.assert_array_equal(hmm.evidence_recursion(np.zeros((3, 2))), np.full((3, 2), .5))

    def test_all_bound_primary_families_accept_the_shared_wrapper(self):
        families = [
            ('Gaussian', 'EKF'), ('Gaussian', 'UKF'), ('GGIW', 'GGIWEKF'),
            ('USGGIW', 'USGGIWEKF'), ('TrajectoryGGIW', 'TrajectoryGGIWEKF'),
            ('TrajectoryUSGGIW', 'TrajectoryUSGGIWEKF')]
        for package_name, filter_name in families:
            with self.subTest(package=package_name, filter=filter_name):
                density = package_name
                wrapped = brewpy.filters.IntentFilter(getattr(brewpy.filters, filter_name)(), estimator(density))
                rfs = brewpy.multiObjectTrackers.PHD(wrapped)
                rfs.set_filter(wrapped)
        with self.assertRaises(TypeError):
            brewpy.filters.IntentFilter(brewpy.filters.EKF(), estimator("GGIW"))

    def test_categorical_merge_aligns_labels_and_preserves_primary_moments(self):
        mixtures = [brewpy.models.GaussianMixture(), brewpy.models.GaussianMixture()]
        for augmented, mixture in enumerate(mixtures):
            a, b = gaussian(0.), gaussian(.2)
            if augmented:
                a.intent = mark([.8, .2]); b.intent = mark([.8, .2], ('b', 'a'))
            mixture.add_component(a, 3.); mixture.add_component(b, 1.)
            mixture.merge(10.)
        plain, augmented = mixtures
        self.assertEqual(augmented.size(), 1)
        self.assertEqual(augmented.weight(0), plain.weight(0))
        np.testing.assert_array_equal(augmented.component(0).mean, plain.component(0).mean)
        np.testing.assert_array_equal(augmented.component(0).covariance, plain.component(0).covariance)
        np.testing.assert_allclose(augmented.component(0).intent.probabilities, [.65, .35], atol=1e-15)

    def test_ggiw_and_spherical_mergers_preserve_categorical_marks(self):
        constructors = [
            (brewpy.models.GGIWMixture, lambda x: brewpy.models.GGIW(
                8., 2., np.array([x, 0., 0., 0.]), np.eye(4), 10., np.eye(2))),
            (brewpy.models.USGGIWMixture, lambda x: brewpy.models.USGGIW(
                8., 2., np.array([np.cos(x), np.sin(x), 0.]), np.zeros(2), np.eye(4), 10., np.eye(2)))]
        for mixture_type, make in constructors:
            mixture = mixture_type()
            a, b = make(0.), make(.01)
            a.intent = mark([.8, .2]); b.intent = mark([.2, .8])
            mixture.add_component(a, 3.); mixture.add_component(b, 1.)
            mixture.merge(10.)
            self.assertEqual(mixture.size(), 1)
            np.testing.assert_allclose(mixture.component(0).intent.probabilities, [.65, .35], atol=1e-15)

    def test_unmarked_birth_and_independent_component_copies(self):
        mixture = brewpy.models.GaussianMixture()
        source = gaussian(0.); source.intent = mark([.8, .2])
        mixture.add_component(source, 3.)
        source.intent = mark([.1, .9])
        np.testing.assert_array_equal(mixture.component(0).intent.probabilities, [.8, .2])
        mixture.add_component(gaussian(.1), 1.)
        mixture.merge(10.)
        np.testing.assert_allclose(mixture.component(0).intent.probabilities, [.725, .275], atol=1e-15)

    def test_invalid_categorical_distributions_are_rejected(self):
        for probabilities, names in [([.8, -.2], ('a', 'b')), ([.8, .8], ('a', 'b')),
                                     ([.8, .2], ('a', 'a')), ([1.], ('a', 'b'))]:
            with self.subTest(probabilities=probabilities, names=names):
                mixture = brewpy.models.GaussianMixture()
                invalid = gaussian(0.); invalid.intent = mark(probabilities, names)
                mixture.add_component(invalid, 1.)
                mixture.add_component(gaussian(.1), 1.)
                with self.assertRaises(ValueError):
                    mixture.merge(10.)

    def test_unmarked_birth_initializes_intent_on_missed_detection(self):
        wrapper = brewpy.filters.IntentFilter(brewpy.filters.EKF(), estimator("Gaussian"))
        rfs = brewpy.multiObjectTrackers.GaussianPHD(wrapper)
        mixture = brewpy.models.GaussianMixture(); mixture.add_component(gaussian(0.), 1.)
        rfs.set_intensity(mixture)
        rfs.set_prob_detection(.8)
        rfs.set_extract_threshold(0.)
        rfs.correct(np.empty((2,0)))
        rfs.cleanup()
        np.testing.assert_allclose(rfs.latest().component(0).intent.probabilities, [.5, .5], atol=1e-15)

    def test_actual_gaussian_filter_rfs_tracking_is_identical(self):
        for filter_type in (brewpy.filters.EKF, brewpy.filters.UKF):
            primary = filter_type()
            primary.set_dynamics(brewpy.dynamics.SingleIntegrator(2))
            primary.set_process_noise(np.eye(4)*.001)
            primary.set_measurement_noise(np.eye(2)*.01)
            primary.set_measurement_jacobian(np.hstack([np.eye(2), np.zeros((2,2))]))
            plain = brewpy.multiObjectTrackers.GaussianPHD(primary)
            wrapped = brewpy.multiObjectTrackers.GaussianPHD(brewpy.filters.IntentFilter(primary, estimator("Gaussian")))
            initial = gaussian(0.); initial.intent = mark([.8, .2])
            mixture = brewpy.models.GaussianMixture(); mixture.add_component(initial, 1.)
            for rfs in (plain, wrapped):
                rfs.set_intensity(mixture)
                rfs.set_prob_detection(.8)
                rfs.set_extract_threshold(0.)
                rfs.set_merge_threshold(10.)
            for frame in range(3):
                for rfs in (plain, wrapped):
                    rfs.predict(frame, 1.)
                    rfs.correct(np.array([[.1*(frame+1)], [0.]]) if frame != 1 else np.empty((2,0)))
                    rfs.cleanup()
                a, b = plain.latest(), wrapped.latest()
                self.assertEqual(a.size(), b.size())
                self.assertGreater(a.size(), 0)
                for i in range(a.size()):
                    self.assertEqual(a.weight(i), b.weight(i))
                    np.testing.assert_array_equal(a.component(i).mean, b.component(i).mean)
                    np.testing.assert_array_equal(a.component(i).covariance, b.component(i).covariance)
                    expected = .5 + .3 * .8**(frame+1)
                    np.testing.assert_allclose(b.component(i).intent.probabilities, [expected, 1-expected], atol=1e-14)

    def test_trajectory_prediction_merge_and_window_rollover(self):
        density = "TrajectoryUSGGIW"
        primary = brewpy.filters.TrajectoryUSGGIWEKF()
        plain = brewpy.multiObjectTrackers.PHD(primary)
        wrapped = brewpy.multiObjectTrackers.PHD(brewpy.filters.IntentFilter(primary, estimator(density)))
        mixture = brewpy.models.TrajectoryUSGGIWMixture()
        for angle, probability, weight in [(0., .8, 2.), (.001, .2, 1.)]:
            inner = brewpy.models.USGGIW(8., 2., np.array([np.cos(angle), np.sin(angle), 0.]),
                np.zeros(2), np.eye(4)*.001, 10., np.eye(2)*.01)
            density = brewpy.models.TrajectoryUSGGIW(4, inner, 2)
            density.intent = mark([probability, 1-probability])
            mixture.add_component(density, weight)
        for rfs in (plain, wrapped):
            rfs.set_intensity(mixture)
            rfs.set_prob_detection(.2)
            rfs.set_extract_threshold(0.)
            rfs.set_merge_threshold(10.)
        for frame in range(4):
            for rfs in (plain, wrapped):
                rfs.predict(frame, 1.)
                rfs.correct(np.empty((3,0)))
                rfs.cleanup()
            a, b = plain.latest(), wrapped.latest()
            self.assertEqual(a.size(), 1)
            self.assertEqual(b.size(), 1)
            self.assertEqual(a.weight(0), b.weight(0))
            for field in ('direction', 'angular_velocity', 'covariance'):
                np.testing.assert_array_equal(getattr(a.component(0).current(), field),
                                              getattr(b.component(0).current(), field))
            expected = .5 + .1 * .8**(frame+1)
            np.testing.assert_allclose(b.component(0).intent.probabilities, [expected, 1-expected], atol=1e-14)


if __name__ == '__main__':
    unittest.main()
