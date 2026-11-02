"""Behavioral regressions for native spherical tracking and report fusion."""
import copy
import unittest
import numpy as np
import brewpy


def state(longitude):
    angle = np.radians(longitude)
    return brewpy.models.USGGIW(12., 2., np.array([np.cos(angle), np.sin(angle), 0.]),
                               np.zeros(2), np.eye(4)*.001, 10., np.eye(2)*.01)


def mixture(*pairs):
    result = brewpy.models.USGGIWMixture()
    for longitude, weight in pairs:
        result.add_component(state(longitude), weight)
    return result


class SphericalTrackingTests(unittest.TestCase):
    def test_existing_phd_types_are_the_only_estimators(self):
        self.assertFalse(hasattr(brewpy.multiObjectTrackers, "Tracker"))
        self.assertFalse(hasattr(brewpy.multiObjectTrackers, "TrajectoryTracker"))
        for tracker in (brewpy.multiObjectTrackers.USGGIWPHD, brewpy.multiObjectTrackers.TrajectoryUSGGIWPHD):
            phd = tracker()
            self.assertFalse(hasattr(phd, "phd"))
            phd.predict(0, 1.)
            phd.correct(np.empty((3,0)))
            phd.cleanup()
            self.assertEqual(phd.latest().size(), 0)

    def test_config_copy_and_unknown_setting(self):
        original = brewpy.multiObjectTrackers.SphericalPHDConfig(max_components=17)
        copied = copy.deepcopy(original)
        copied.max_components = 4
        self.assertEqual(original.max_components, 17)
        with self.assertRaises(TypeError):
            brewpy.multiObjectTrackers.SphericalPHDConfig(misspelled_threshold=1.)

    def test_empty_observations_cluster_birth_and_visibility(self):
        tracker = brewpy.multiObjectTrackers.USGGIWPHD(brewpy.multiObjectTrackers.SphericalPHDConfig(extract_threshold=.01))
        empty = np.empty((0,2))
        self.assertEqual(tracker.step(empty, 1.), [])
        self.assertEqual(tracker.mixture().size(), 0)
        points = np.array([[0.,0.], [.1,0.], [0.,.1], [-.1,0.], [0.,-.1]])
        tracker.step(points, 1., observer_lonlat=np.array([[0.,0.]]))
        self.assertEqual(tracker.mixture().size(), 1)
        self.assertEqual(tracker.report_mixture(np.array([[0.,0.]])).size(), 1)
        self.assertEqual(tracker.report_mixture(np.array([[90.,0.]])).size(), 0)
        self.assertEqual(tracker.report_mixture(empty).size(), 0)
        tracker.step(points, 1., observer_lonlat=np.array([[0.,0.]]))
        self.assertEqual(tracker.mixture().size(), 1)

    def test_fusion_uses_max_weight_and_one_n_source_gci(self):
        fusion = brewpy.utils.fusion.ReportFusion(6.)
        inputs = [mixture((x,w)) for x,w in [(0.,.2), (2.,.8), (4.,.4)]]
        rings, pooled = fusion.fuse_rings([inputs, []])
        self.assertEqual([r.size() for r in rings], [1,0])
        self.assertEqual(pooled.size(), 1)
        self.assertEqual(pooled.weight(0), .8)
        # The strongest report is the reference; each source contributes equally.
        oracle = brewpy.utils.fusion.gci([mixture((2.,1.)), mixture((0.,1.)), mixture((4.,1.))],
                                  [1/3]*3, 0.)
        np.testing.assert_allclose(pooled.component(0).direction, oracle.component(0).direction,
                                   rtol=0., atol=1e-14)
        np.testing.assert_allclose(pooled.component(0).covariance, oracle.component(0).covariance,
                                   rtol=0., atol=1e-14)
        self.assertEqual(fusion.fuse_rings([])[1].size(), 0)

    def test_fusion_gate_is_not_transitive_and_ties_keep_input_order(self):
        fused = brewpy.utils.fusion.ReportFusion(6.).reduce(mixture((0.,.5), (5.,.5), (10.,.5)))
        self.assertEqual(fused.size(), 2)
        self.assertAlmostEqual(np.degrees(np.arctan2(*fused.component(1).direction[1::-1])), 10.)

    def test_trajectory_window_rollover_and_extracted_cardinality(self):
        config = brewpy.multiObjectTrackers.SphericalPHDConfig(extract_threshold=.01)
        tracker = brewpy.multiObjectTrackers.TrajectoryUSGGIWPHD(config, .2, .04, .0001,
                                                  window_size=3, velocity_sigma_deg_min=.01)
        observers = np.array([[0.,0.], [90.,0.]])
        self.assertEqual(tracker.step(mixture(), 1., observers, [0,1]), [])
        for frame in range(8):
            states = tracker.step(mixture((0.,.8)), 1., observers, [0,1])
            self.assertEqual(len(states), 1)
            self.assertTrue(np.isfinite(states[0].covariance).all())
        posterior = tracker.latest()
        self.assertLessEqual(posterior.component(0).window_size(), 3)
        history = tracker.track_histories()[0]
        self.assertGreater(history.shape[0], 3)
        np.testing.assert_allclose(history[-1], states[0].direction)
        self.assertAlmostEqual(tracker.extracted_cardinality(), sum(posterior.weight(i) for i in range(posterior.size())))

    def test_dedup_selects_strongest_without_removing_posterior_hypotheses(self):
        config = brewpy.multiObjectTrackers.SphericalPHDConfig(extract_threshold=.01, merge_threshold=0.,
                                          prob_detection=0., prob_survive=1.)
        tracker = brewpy.multiObjectTrackers.TrajectoryUSGGIWPHD(config, .2, .04, .0001, dedup_deg=8.)
        prior = brewpy.models.TrajectoryUSGGIWMixture()
        prior.add_component(brewpy.models.TrajectoryUSGGIW(4, state(0.), 8), .8)
        prior.add_component(brewpy.models.TrajectoryUSGGIW(4, state(5.), 8), .9)
        tracker.set_intensity(prior)
        states = tracker.step(mixture(), 1., np.empty((0,2)))
        self.assertEqual(len(states), 1)
        np.testing.assert_allclose(states[0].direction, state(5.).direction, atol=1e-15)
        self.assertEqual(tracker.latest().size(), 2)
        self.assertAlmostEqual(tracker.extracted_cardinality(), 1.7)

    def test_history_arrays_are_snapshots_across_native_updates(self):
        tracker = brewpy.multiObjectTrackers.TrajectoryUSGGIWPHD(
            brewpy.multiObjectTrackers.SphericalPHDConfig(extract_threshold=.01), .2, .04, .0001)
        observers = np.array([[0.,0.]])
        tracker.step(mixture((0.,.8)), 1., observers)
        history = tracker.current_extracted_histories[0]
        # A parent reference would keep the tracker alive but not its replaced matrices.
        self.assertNotIsInstance(history.base, brewpy.multiObjectTrackers.TrajectoryUSGGIWPHD)
        snapshot = history.copy()
        tracker.step(mixture((1.,.8)), 1., observers)
        np.testing.assert_array_equal(history, snapshot)

    def test_existing_threshold_setter_controls_complete_step(self):
        phd = brewpy.multiObjectTrackers.USGGIWPHD(brewpy.multiObjectTrackers.SphericalPHDConfig(extract_threshold=.01))
        points = np.array([[0.,0.], [.1,0.], [0.,.1], [-.1,0.], [0.,-.1]])
        self.assertEqual(len(phd.step(points, 1.)), 1)
        phd.set_extract_threshold(2.)
        self.assertEqual(phd.step(points, 1.), [])
        self.assertEqual(phd.mixture().size(), 0)

    def test_total_and_extracted_cardinality_remain_distinct(self):
        config = brewpy.multiObjectTrackers.SphericalPHDConfig(extract_threshold=.85, merge_threshold=0.,
                                       prob_detection=0., prob_survive=1.)
        phd = brewpy.multiObjectTrackers.TrajectoryUSGGIWPHD(config, .2, .04, .0001)
        prior = brewpy.models.TrajectoryUSGGIWMixture()
        prior.add_component(brewpy.models.TrajectoryUSGGIW(4, state(0.), 8), .8)
        prior.add_component(brewpy.models.TrajectoryUSGGIW(4, state(20.), 8), .9)
        phd.set_intensity(prior)
        phd.step(mixture(), 1., np.empty((0,2)))
        self.assertAlmostEqual(phd.expected_cardinality(), 1.7)
        self.assertAlmostEqual(phd.extracted_cardinality(), .9)


if __name__ == '__main__':
    unittest.main()
