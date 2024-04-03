import numpy as np
import pandas as pd
from astropy.coordinates import ICRS, Galactic
from astropy import units as u
from modules.cluster_sampler import ClusterSampler
from modules.compute_uncertainties import GaiaUncertainties
from modules.compute_observables import ErrorBase


# ------ Path files ------
fpath = '/Users/ratzenboe/Documents/work/code/SF-Retreat2024/isochrone_files/'
parsec_files = fpath + 'parsec_files/'
baraffe_files = fpath + 'baraffe_files/'
fname_spline_csv = 'LogErrVsMagSpline.csv'
fname_astrometric_corr = 'astrometric_corr.npz'
# -----------------------

class ClusterSim:
    def __init__(self, mu, cov, mass, logAge,
                 feh=0., f_bin=0.3, parsec_folder=parsec_files, baraffe_folder=baraffe_files,
                 M_G_threshold=5, spline_csv=fname_spline_csv, astrometric_corr=fname_astrometric_corr):
        self.spline_csv = spline_csv
        self.astrometric_corr = astrometric_corr
        # Instantiate the ClusterSampler class
        self.cluster_sampler = ClusterSampler(
            mu, cov, mass, logAge, feh, f_bin, parsec_folder, baraffe_folder, M_G_threshold
        )
        self.cluster_sampler.simulate_cluster()
        # Instantiate the GaiaUncertainties class and the ErrorBase class
        self.unc_obj = GaiaUncertainties(cluster_object=self.cluster_sampler, spline_csv=self.spline_csv)
        # Instantiate the ErrorBase class
        self.errs = ErrorBase(unc_obj=self.unc_obj, astrometric_fname=self.astrometric_corr)
        # hardcoded for now, but can be changed to be set by the user it
        self.features_returned = [
            'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity',
            'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'M_G', 'X',
            'Y', 'Z', 'U', 'V', 'W'
        ]

    def data_observed(self, **kwargs):
        if kwargs:
            self.cluster_sampler.set_cluster_params(**kwargs)
            self.cluster_sampler.simulate_cluster()
            self.unc_obj.new_cluster(self.cluster_sampler)
            # Instantiate the ErrorBase class
            self.errs = ErrorBase(self.unc_obj, self.astrometric_corr)

        df_obs = self.errs.convolve()
        # Compute observed absolute magnitude
        df_obs['M_G'] = df_obs['phot_g_mean_mag'] - 5 * np.log10(1000 / df_obs['parallax']) + 5
        # XYZ+UVW
        df_cart = self.spher2cart(df_obs[['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']].values)
        df_obs = pd.concat([df_obs, df_cart], axis=1)
        return df_obs[self.features_returned]

    def data_true(self):
        df_cart = pd.DataFrame(self.errs.set_X(), columns=self.errs.features)
        df_true = pd.concat([self.cluster_sampler.data_phot, self.cluster_sampler.X_gal, df_cart], axis=1)
        return df_true  #[self.features_returned + ['logg', 'teff', 'mass', 'is_binary']]
        # return self.cluster_sampler.data_phot

    @staticmethod
    def spher2cart(data):
        ra, dec, parallax, pmra, pmdec, rv = data.T
        dist = 1000 / parallax
        dist[dist < 0] = 1e4
        c = ICRS(
            ra=ra * u.deg, dec=dec * u.deg, distance=dist * u.pc,
            pm_ra_cosdec=pmra * u.mas / u.yr,
            pm_dec=pmdec * u.mas / u.yr,
            radial_velocity=rv * u.km / u.s,
        )
        c = c.transform_to(Galactic())
        c.representation_type = 'cartesian'
        X = np.vstack([c.u.value, c.v.value, c.w.value, c.U.value, c.V.value, c.W.value]).T
        df = pd.DataFrame(X, columns=['X', 'Y', 'Z', 'U', 'V', 'W'])
        return df