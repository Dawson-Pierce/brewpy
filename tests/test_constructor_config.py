"""Constructor configuration must match the existing setter path exactly."""
import unittest
import numpy as np
from brewpy import dynamics, models, filters, multiObjectTrackers, utils


class ConstructorConfigTests(unittest.TestCase):
    def run_pose(self, keywords):
        points = np.random.default_rng(42).normal(size=(3, 40))
        library = utils.registration.TemplateLibrary()
        template_id = library.add(points)
        impulse = models.ImpulsePrior(.2, .01)
        motion = dynamics.SingleIntegrator(3)
        motion.set_impulse_model(dynamics.ImpulseModel(impulse, body_frame=True))
        settings = dict(
            dynamics=motion, template_library=library,
            icp=utils.registration.PointToPointIcp(),
            process_noise=np.eye(6)*1e-5,
            measurement_noise=np.eye(6)*.01,
            measurement_jacobian=np.hstack([np.eye(3), np.zeros((3,3))]),
            rotation_process_noise=np.eye(3)*1e-5,
            surface_outlier_probability=0., pose_return_cap=25,
        )
        primary = filters.TmEkf(**settings) if keywords else filters.TmEkf()
        if not keywords:
            for name, value in settings.items(): getattr(primary, 'set_'+name)(value)
        intent = filters.IntentEstimator([
            models.IntentHypothesis('coast', models.PointingFiringModel(
                models.PointingRule.Fixed, impulse, firing_probability=0.)),
            models.IntentHypothesis('approach', models.PointingFiringModel(
                models.PointingRule.Radial, impulse, radial_sign=-1.)),
        ], dt=1.)
        wrapped = filters.IntentFilter(primary, intent)
        births = models.TemplatePoseMixture()
        births.add_component(models.TemplatePose(
            np.array([100.,0,0,0,0,0]), np.eye(9)*.01,
            template_id, [0,1,2], np.eye(3), 0), .8)
        settings = dict(birth_model=births, prob_detection=.95, prob_survive=.99,
                        clutter_rate=1., clutter_density=1e-6,
                        prune_threshold=0., extract_threshold=0., merge_threshold=4.,
                        merge_rotation_threshold=.1, max_components=20,
                        gate_threshold=100., cluster_object=utils.clustering.DBSCAN(
                            8.,3,[0,1,2],[1.,1.,1.]))
        phd = multiObjectTrackers.PHD(wrapped, **settings) if keywords else multiObjectTrackers.PHD(wrapped)
        if not keywords:
            for name, value in settings.items(): getattr(phd, 'set_'+name)(value)
        # Constructor clones births; changing the caller's mixture must have no effect.
        births.add_component(models.TemplatePose(
            np.zeros(6), np.eye(9), template_id, [0,1,2]), 10.)
        output = []
        for frame in range(3):
            if frame: phd.set_birth_model(models.TemplatePoseMixture())
            phd.predict(frame, 1.)
            phd.correct(points + np.array([[100.+frame*.01],[0.],[0.]]))
            phd.cleanup()
            mixture = phd.latest()
            output.append(np.array([mixture.weight(i) for i in range(mixture.size())]))
            for i in range(mixture.size()):
                d = mixture.component(i)
                output.extend([d.mean.copy(), d.covariance.copy(), d.rotation.copy(),
                               d.intent.probabilities.copy(), np.array(d.firing.impulse_probability)])
        return output

    def test_constructor_and_setters_give_identical_pose_intent_results(self):
        a, b = self.run_pose(False), self.run_pose(True)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b): np.testing.assert_array_equal(x, y)

    def test_typed_and_generic_phd_accept_zero_and_omitted_values(self):
        primary = filters.EKF()
        for constructor in (multiObjectTrackers.PHD, multiObjectTrackers.GaussianPHD):
            constructor(primary, max_history=1)
            births = models.GaussianMixture()
            births.add_component(models.Gaussian(np.zeros(2), np.eye(2)), .5)
            phd = constructor(primary, intensity=births, extract_threshold=0.,
                              prune_threshold=0., merge_threshold=0.)
            phd.cleanup()
            self.assertEqual(phd.latest().size(), 1)
            self.assertEqual(phd.latest().weight(0), .5)

    def test_invalid_keywords_types_and_probabilities(self):
        with self.assertRaisesRegex(TypeError, 'Unknown TmEkf'):
            filters.TmEkf(process_noize=np.eye(6))
        with self.assertRaisesRegex(TypeError, 'Unknown PHD'):
            multiObjectTrackers.PHD(filters.EKF(), prune_treshold=.1)
        with self.assertRaises((TypeError, RuntimeError)):
            filters.TmEkf(process_noise='bad')
        with self.assertRaises(ValueError):
            multiObjectTrackers.PHD(filters.EKF(), prob_detection=1.1)

    def test_trajectory_factories_share_keyword_configuration(self):
        for name in ('TrajectoryGGIW', 'TrajectoryUSGGIW'):
            primary = getattr(filters, name+'EKF')()
            factory = getattr(multiObjectTrackers, name+'PHD')
            for constructor in (factory, multiObjectTrackers.PHD):
                phd = constructor(primary, prob_detection=0., max_history=1,
                                  extract_threshold=None, birth_model=None)
                phd.cleanup()
                self.assertEqual(phd.latest().size(), 0)


if __name__ == '__main__': unittest.main()
