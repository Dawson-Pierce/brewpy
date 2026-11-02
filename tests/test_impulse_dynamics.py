"""Impulse dynamics are shared across primary filters and RFS composition."""
import unittest
import numpy as np
from brewpy import dynamics, models, filters, multiObjectTrackers


class ImpulseDynamicsTests(unittest.TestCase):
    def test_euclidean_filters_share_the_same_motion_model(self):
        prior = models.ImpulsePrior(.2, .01)
        motion = dynamics.SingleIntegrator(2)
        motion.set_impulse_model(dynamics.ImpulseModel(prior, axis=[1., 0.], velocity_indices=[2, 3]))
        for primary, density, mixture_type in [
            (filters.EKF(), models.Gaussian(np.zeros(4), np.eye(4)), models.GaussianMixture),
            (filters.UKF(), models.Gaussian(np.zeros(4), np.eye(4)), models.GaussianMixture),
            (filters.GGIWEKF(), models.GGIW(8.,2.,np.zeros(4),np.eye(4),10.,np.eye(2)), models.GGIWMixture),
        ]:
            with self.subTest(filter=type(primary).__name__):
                primary.set_dynamics(motion)
                primary.set_process_noise(np.zeros((4,4)))
                phd = multiObjectTrackers.PHD(primary)
                phd.set_prob_survive(1.)
                mixture = mixture_type(); mixture.add_component(density, .8)
                phd.set_intensity(mixture)
                phd.predict(0,2.)
                # latest() is extracted output; expose the posterior via zero-threshold cleanup.
                phd.set_extract_threshold(0.)
                phd.set_prune_threshold(0.)
                phd.set_merge_threshold(0.)
                phd.cleanup()
                result = phd.latest()
                self.assertEqual(result.size(),2)
                components = sorted([(result.weight(i), result.component(i)) for i in range(2)], reverse=True, key=lambda x:x[0])
                np.testing.assert_allclose([w for w,_ in components], [.792,.008], atol=1e-14)
                a,b = [d for _,d in components]
                np.testing.assert_allclose(b.mean-a.mean,[.2,0,.1,0],atol=1e-14)
                response=np.array([2.,0,1.,0])
                np.testing.assert_allclose(b.covariance-a.covariance,(.2**2/12)*np.outer(response,response),atol=1e-14)

    def test_prior_lives_on_dynamics(self):
        prior=models.ImpulsePrior(.2,.01)
        motion=dynamics.ClohessyWiltshire(.001)
        self.assertIsNone(motion.impulse_model)
        motion.set_impulse_model(dynamics.ImpulseModel(prior,body_frame=True))
        self.assertTrue(motion.impulse_model.body_frame)
        self.assertEqual(motion.impulse_model.prior.maximum,.2)
        self.assertEqual(motion.impulse_model.prior.probability,.01)
        self.assertFalse(hasattr(filters.TmEkf(),'set_impulse_prior'))

    def test_invalid_axis_and_indices_are_rejected(self):
        prior=models.ImpulsePrior(.2,.01)
        for kwargs in [dict(axis=[0,0,0]),dict(velocity_indices=[3,3,4]),
                       dict(axis=[1,0],velocity_indices=[2,3],body_frame=True)]:
            with self.assertRaises(ValueError): dynamics.ImpulseModel(prior,**kwargs)


if __name__ == '__main__': unittest.main()
