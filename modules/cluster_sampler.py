import numpy as np
import pandas as pd
from modules.isochrones import ClusterPhotometry
from astropy.coordinates import ICRS, Galactic, SkyCoord
from astropy import units as u


class ClusterSampler(ClusterPhotometry):
    def __init__(self, mu, cov, mass, logAge, feh, f_bin, parsec_folder, baraffe_folder, M_G_threshold=5):
        super().__init__(parsec_folder, baraffe_folder, M_G_threshold)
        # super().__init__(astrometric_fname, rv_fname)
        self.cluster_mu = mu
        self.cluster_cov = cov
        self.cluster_mass = mass
        self.cluster_logAge = logAge
        self.cluster_feh = feh
        self.f_bin_total = f_bin
        # Source data
        self.cluster_size = None
        # Simpulated cluster data
        self.skycoord = None
        self.X_gal = None
        self.X_icrs = None
        # Observed data
        # self.X_obs_gal = None
        # self.X_obs_icrs = None

    def set_cluster_params(self, **kwargs):
        mu = kwargs.get('mu', self.cluster_mu)
        cov = kwargs.get('cov', self.cluster_cov)
        mass = kwargs.get('mass', self.cluster_mass)
        logAge = kwargs.get('logAge', self.cluster_logAge)
        feh = kwargs.get('feh', self.cluster_feh)
        f_bin = kwargs.get('f_bin', self.f_bin_total)
        # Set parameters
        self.cluster_mu = mu
        self.cluster_cov = cov
        self.cluster_mass = mass
        self.cluster_logAge = logAge
        self.cluster_feh = feh
        self.f_bin_total = f_bin

    @staticmethod
    def skycoord_from_galactic(data):
        X, Y, Z, U, V, W = data.T
        c = Galactic(
            u=X * u.pc, v=Y * u.pc, w=Z * u.pc,
            U=U * u.km / u.s, V=V * u.km / u.s, W=W * u.km / u.s,
            representation_type="cartesian",
            # Velocity representation
            differential_type="cartesian",
        )
        c_icrs = c.transform_to(ICRS())
        c_icrs.representation_type = 'spherical'
        skycoords = SkyCoord(c_icrs)
        return skycoords

    def simulate_cluster(self):
        __data = self.make_photometry(self.cluster_mass, self.cluster_logAge, self.cluster_feh, self.f_bin_total)
        self.cluster_size = __data.shape[0]
        # Create cluster data in XYZ+UVW
        self.X_gal = pd.DataFrame(
            np.random.multivariate_normal(self.cluster_mu, self.cluster_cov, self.cluster_size),
            columns=['X', 'Y', 'Z', 'U', 'V', 'W']
        )
        # Convert to ICRS
        self.skycoord = self.skycoord_from_galactic(self.X_gal.values)

        # Simulate errors
        # self.simulate_errors(self.cluster_size)
        # # Convolve the data with measurement uncertainties
        # self.X_obs_icrs = self.convolve(self.X_icrs)
        # self.X_obs_gal = spher2cart(self.X_obs_icrs)
        return