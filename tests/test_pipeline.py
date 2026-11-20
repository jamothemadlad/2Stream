import numpy as np
import pytest
from twostream.io import C, momentum_to_velocity, bin_particles
from twostream.forecast import ballistic_transport, check_split


def test_physical_velocity_from_relativistic_si_momentum():
    mass=np.array([9.1093837139e-31])
    momentum=np.array([[mass[0]*C,0.,0.]])
    velocity,gamma=momentum_to_velocity(momentum,mass)
    np.testing.assert_allclose(gamma,np.sqrt(2))
    np.testing.assert_allclose(velocity[0,0],C/np.sqrt(2))


def test_histogram_accounts_for_excluded_weight():
    hist,lost=bin_particles(np.array([.25,.75,.5]),np.array([-.1,.1,2.]),
                           np.array([1.,2.,1.]),np.array([0.,.5,1.]),np.array([-1.,0.,1.]))
    assert lost==.25
    assert hist.sum()==.75
    assert hist[0,0]==.25
    assert hist[1,1]==.5


def test_periodic_ballistic_transport_preserves_mass_and_direction():
    maps=np.zeros((1,2,4,1));maps[0,:,3,0]=1
    z=np.arange(5,dtype=float)
    velocity=np.array([0.,2/C])
    predicted=ballistic_transport(maps,z,velocity,1.)
    np.testing.assert_allclose(predicted[0,:,0,0],1)
    np.testing.assert_allclose(predicted.sum(axis=(-2,-1)),maps.sum(axis=(-2,-1)))


def test_fractional_ballistic_shift_is_conservative():
    maps=np.zeros((1,2,4,1));maps[0,:,0,0]=1
    predicted=ballistic_transport(maps,np.arange(5,dtype=float),np.array([0.,1/C]),1.)
    np.testing.assert_allclose(predicted[0,:,0,0],.5)
    np.testing.assert_allclose(predicted[0,:,1,0],.5)
    np.testing.assert_allclose(predicted.sum(axis=(-2,-1)),1)


def test_run_split_rejects_seed_leakage():
    with pytest.raises(ValueError,match='more than one split'):
        check_split({1:None,2:None,3:None},[1],[2],[1,3])
