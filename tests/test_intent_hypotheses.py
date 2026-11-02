"""Application-supplied hypotheses must determine native likelihood behavior."""
import unittest
import numpy as np
import brewpy

native = brewpy.models


class IntentHypothesisTests(unittest.TestCase):
    def test_saved_records_feed_native_history_without_report_wrappers(self):
        from types import SimpleNamespace
        record = SimpleNamespace(
            mean=np.zeros(6), covariance=np.eye(9), template_id=0,
            rotation=np.eye(3), mode_id=0, w=.8, intent_posterior=None,
            impulse_observation=.1, impulse_variance=.001,
            impulse_log_zero=-5., impulse_log_positive=-1.,
            impulse_probability=.7, impulse_magnitude=.03, impulse_std=.02,
            impulse_lower=0., impulse_upper=.1)
        estimator = brewpy.filters.IntentEstimator([native.IntentHypothesis(
            'coast', native.PointingFiringModel(native.PointingRule.Fixed))])
        np.testing.assert_array_equal(estimator.posterior_history([record, record]), [[1.], [1.]])
        self.assertIsNone(record.intent_posterior)
        self.assertEqual(record.impulse_probability, .7)
        for removed in ['reports', 'step', 'reset', 'gate']:
            self.assertFalse(hasattr(estimator, removed))
        self.assertFalse(hasattr(native, 'IntentPose'))
        self.assertFalse(hasattr(native, 'IntentEstimate'))

    def test_shared_impulse_prior_preserves_inference(self):
        impulse = native.ImpulsePrior(maximum=.2, probability=.01)
        with self.assertRaises(AttributeError):
            impulse.maximum = .4
        for maximum, probability in [(0, .1), (.2, 0), (.2, 1), (float('nan'), .1)]:
            with self.assertRaises(ValueError):
                native.ImpulsePrior(maximum, probability)
        shared = [native.IntentHypothesis(name, native.PointingFiringModel(
            native.PointingRule.Fixed, impulse, firing_probability=p))
            for name, p in [('coast', 0.), ('maneuver', None)]]
        legacy = [native.IntentHypothesis(name, native.PointingFiringModel(
            native.PointingRule.Fixed, impulse_max=.2, firing_probability=p))
            for name, p in [('coast', 0.), ('maneuver', .01)]]
        self.assertEqual(shared[1].model.firing_probability, impulse.probability)
        self.assertEqual(shared[0].model.firing_probability, 0.)
        self.assertTrue(all(h.model.impulse_max == impulse.maximum for h in shared))
        estimators = [brewpy.filters.IntentEstimator(shared, dt=30.),
                      brewpy.filters.IntentEstimator(legacy, dt=30.)]
        for kwargs in ({'impulse': impulse}, {'impulse_max': .2}, {'impulse_probability': .01}):
            with self.assertRaises(TypeError):
                brewpy.filters.IntentEstimator(shared, **kwargs)
        self.assertFalse(hasattr(estimators[0], 'impulse_posterior'))
        histories = []
        for estimator in estimators:
            previous = brewpy.models.TemplatePose(np.array([100., 0, 0, 0, 0, 0]), np.eye(9), 0, [0, 1, 2])
            current = brewpy.models.TemplatePose(np.array([101., 0, 0, 0, 0, 0]), np.eye(9), 0, [0, 1, 2])
            current.set_impulse_evidence(.1, .0001)
            report = native.FiringEstimate()
            report.impulse_probability = .7
            report.impulse_magnitude = .03
            report.impulse_std = .02
            report.impulse_lower = 0.
            report.impulse_upper = .1
            current.firing = report
            estimator.correct(previous)
            estimator.correct(current, previous)
            mark = current.intent
            histories.append(mark.probabilities)
            for field in ['probability', 'magnitude', 'std', 'lower', 'upper']:
                self.assertEqual(getattr(current.firing, 'impulse_' + field),
                                 getattr(report, 'impulse_' + field))
        np.testing.assert_array_equal(*histories)
        logs = np.random.default_rng(42).normal(size=(12, 2))
        np.testing.assert_array_equal(*(e.evidence_recursion(logs) for e in estimators))

    def test_pointing_rules(self):
        position = np.array([3., 4., 0.])
        definitions = (
            (native.PointingFiringModel(native.PointingRule.Fixed, axis=[0, 2, 0]), [0, 1, 0]),
            (native.PointingFiringModel(native.PointingRule.Radial, radial_sign=-1), [-.6, -.8, 0]),
            (native.PointingFiringModel(native.PointingRule.Tangential, axis=[0, 0, 1]), [-.8, .6, 0]),
        )
        for hypothesis, expected in definitions:
            np.testing.assert_allclose(hypothesis.boresight(position), expected, atol=1e-15)

    def test_firing_probability_controls_likelihood_independently_of_label(self):
        hypotheses = [native.IntentHypothesis(name, native.PointingFiringModel(
                          native.PointingRule.Fixed, firing_probability=p))
                      for name, p in [('custom idle', 0.), ('custom maneuver', .3), ('coast', 1.)]]
        estimator = brewpy.filters.IntentEstimator(hypotheses)
        evidence = [-2., -1.]
        actual = [h.model.impulse_log_likelihood(.01, .001, evidence) for h in estimator.hypotheses]
        expected = [evidence[0], np.logaddexp(np.log(.7)-2., np.log(.3)-1.), evidence[1]]
        np.testing.assert_allclose(actual, expected, atol=1e-15)
        self.assertEqual(estimator.objectives, [h.name for h in hypotheses])

    def test_arbitrary_count_and_order(self):
        hypotheses = [native.IntentHypothesis(f'policy {i}', native.PointingFiringModel(
                          native.PointingRule.Fixed, firing_probability=i/6)) for i in range(7)]
        estimator = brewpy.filters.IntentEstimator(hypotheses)
        reordered = brewpy.filters.IntentEstimator(hypotheses[::-1])
        a = [h.model.impulse_log_likelihood(.01, .001) for h in estimator.hypotheses]
        b = [h.model.impulse_log_likelihood(.01, .001) for h in reordered.hypotheses]
        np.testing.assert_array_equal(b, a[::-1])
        evidence = np.random.default_rng(42).normal(size=(12, 7))
        expected = np.full_like(evidence, 1/7)
        transition = np.full((7, 7), .1/6)
        np.fill_diagonal(transition, .9)
        for i in range(1, len(evidence)):
            weights = (transition.T @ expected[i-1]) * np.exp(evidence[i])
            expected[i] = weights / weights.sum()
        np.testing.assert_allclose(estimator.evidence_recursion(evidence), expected, atol=1e-14)

    def test_one_hypothesis_and_new_density(self):
        estimator = brewpy.filters.IntentEstimator([native.IntentHypothesis(
            'custom', native.PointingFiringModel(native.PointingRule.Fixed))])
        density = brewpy.models.TemplatePose(np.zeros(6), np.eye(9), 0, [0, 1, 2])
        estimator.correct(density)
        self.assertEqual(density.intent.objectives, ['custom'])
        np.testing.assert_array_equal(density.intent.probabilities, [1.])

    def test_configuration_errors_are_python_exceptions(self):
        for arguments in ({'axis': [0, 0, 0]}, {'radial_sign': 0.},
                          {'firing_probability': -1.}, {'firing_probability': 2.},
                          {'firing_probability': float('nan')}, {'gain_max': 0.},
                          {'attitude_sigma': 0.}, {'holding_probability': 2.}, {'impulse_max': -1.}):
            settings = dict(pointing=native.PointingRule.Fixed)
            settings.update(arguments)
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                native.PointingFiringModel(**settings)
        with self.assertRaises(ValueError):
            brewpy.filters.IntentEstimator([])
        model = native.PointingFiringModel(native.PointingRule.Fixed)
        with self.assertRaises(ValueError):
            native.IntentHypothesis('', model)
        with self.assertRaises(ValueError):
            native.IntentHypothesis('missing', None)
        policy = native.IntentHypothesis('same', model)
        with self.assertRaises(ValueError):
            brewpy.filters.IntentEstimator([policy, policy])

    def test_native_model_lifetime_and_density_update(self):
        import gc
        models = [native.PointingFiringModel(native.PointingRule.Fixed, firing_probability=p,
                                             impulse_max=.2) for p in [0., .01]]
        estimator = brewpy.filters.IntentEstimator([native.IntentHypothesis(str(i), m) for i, m in enumerate(models)])
        del models
        gc.collect()
        previous = brewpy.models.TemplatePose(np.array([100., 0, 0, 0, 0, 0]), np.eye(9), 0, [0, 1, 2])
        current = brewpy.models.TemplatePose(np.array([101., 0, 0, 0, 0, 0]), np.eye(9), 0, [0, 1, 2])
        current.set_impulse_evidence(.1, .0001)
        estimator.correct(previous)
        logs = np.log(previous.intent.probabilities)
        for i, hypothesis in enumerate(estimator.hypotheses):
            self.assertIsInstance(hypothesis.model, native.TemplatePosePredictiveBehaviorModel)
            for term in hypothesis.model.log_evidence(current, previous, dt=30.):
                logs[i] += term
        expected = np.exp(logs - np.max(logs))
        expected /= expected.sum()
        estimator.correct(current, previous)
        np.testing.assert_allclose(current.intent.probabilities, expected, atol=1e-14)
        np.testing.assert_array_equal(current.mean, [101., 0, 0, 0, 0, 0])
        with self.assertRaises(AttributeError):
            estimator.hypotheses[0].model.firing_probability = .7
        with self.assertRaises(TypeError):
            native.TemplatePosePredictiveBehaviorModel()


if __name__ == '__main__':
    unittest.main()
