"""Births compete with continuations without erasing motion or inventing impulses."""
import unittest
import numpy as np
from brewpy import dynamics, filters, models, multiObjectTrackers, utils


class BirthContinuationTests(unittest.TestCase):
    def test_late_arrivals_velocity_learning_and_firing(self):
        rng = np.random.default_rng(311)
        points = rng.normal(size=(3, 90)) * np.array([[2.], [1.], [.5]])
        library = utils.registration.TemplateLibrary()
        template_id = library.add(points)
        prior = models.ImpulsePrior(.2, .01)
        motion = dynamics.SingleIntegrator(3)
        motion.set_impulse_model(dynamics.ImpulseModel(prior, body_frame=True))
        primary = filters.TmEkf(
            dynamics=motion, template_library=library,
            icp=utils.registration.PointToPointIcp(),
            process_noise=np.diag([.2**2]*3 + [.0002**2]*3),
            measurement_noise=.01*np.eye(6), rotation_process_noise=1e-5*np.eye(3))
        intent = filters.IntentEstimator([
            models.IntentHypothesis('coast', models.PointingFiringModel(
                models.PointingRule.Fixed, prior, firing_probability=0.)),
            models.IntentHypothesis('maneuver', models.PointingFiringModel(
                models.PointingRule.Fixed, prior)),
        ], dt=30.)
        phd = multiObjectTrackers.PHD(filters.IntentFilter(primary, intent),
            prob_detection=1., prob_survive=.99, clutter_rate=1., clutter_density=1e-6,
            prune_threshold=1e-5, extract_threshold=0., merge_threshold=4.,
            merge_rotation_threshold=.2, max_components=80, gate_threshold=100.,
            cluster_object=utils.clustering.DBSCAN(8., 3, [0,1,2], [1.,1.,1.]))
        position = np.array([100., 220.])
        velocity = np.array([.025, -.03])
        for frame in range(18):
            if frame == 12:
                velocity[0] += .1
            if frame:
                position += 30.*velocity
            visible = ([0] if frame >= 3 else []) + ([1] if frame >= 7 and frame != 15 else [])
            births = models.TemplatePoseMixture()
            clouds = []
            for obj in visible:
                cloud = points + np.array([[position[obj]], [0.], [0.]]) + rng.normal(0., .01, points.shape)
                clouds.append(cloud)
                birth = models.TemplatePose(np.r_[cloud.mean(axis=1)-points.mean(axis=1), np.zeros(3)],
                    np.diag([12.**2]*3 + [.2**2]*3 + [(np.pi/4)**2]*3), template_id, [0,1,2])
                birth.mark_pca_aligned()
                births.add_component(birth, .4)
            phd.set_birth_model(births)
            phd.predict(frame, 0. if frame == 0 else 30.)
            phd.correct(np.hstack(clouds) if clouds else np.empty((3,0)))
            phd.cleanup()
            posterior = phd.latest()
            for obj in visible:
                candidates = [i for i in range(posterior.size())
                              if abs(posterior.component(i).mean[0]-position[obj]) < 1.]
                self.assertTrue(candidates, (frame, obj))
                mass = sum(posterior.weight(i) for i in candidates)
                self.assertAlmostEqual(mass, 1., delta=1e-3)
                best = max(candidates, key=posterior.weight)
                report = posterior.component(best)
                probability = report.firing.impulse_probability
                arrival = (frame, obj) in ((3,0), (7,1), (16,1))
                if arrival:
                    self.assertTrue(np.isnan(probability), (frame, obj, probability))
                elif (frame, obj) == (12,0):
                    self.assertGreater(probability, .5)
                else:
                    self.assertLess(probability, .5, (frame, obj, probability))
                if frame >= (6 if obj == 0 else 10) and not arrival:
                    self.assertAlmostEqual(report.mean[3], velocity[obj], delta=.015)


if __name__ == '__main__':
    unittest.main()

