import numpy as np
import pandas as pd
from pygaia.errors.spectroscopic import radial_velocity_uncertainty
from pygaia.errors.astrometric import parallax_uncertainty
import scipy.interpolate as interpolate
from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG
from gaiaunlimited.scanninglaw import GaiaScanningLaw
from modules.cluster_sampler import ClusterSampler


class Edr3LogMagUncertainty:
    """
    Estimate the log(mag) vs mag uncertainty for G, G_BP, G_RP based on Gaia EDR3 photometry.
    """

    def __init__(self, spline_csv, n_obs=200):
        """Usage
        >>> u = Edr3LogMagUncertainty('LogErrVsMagSpline.csv')
        >>> gmags = np.array([5, 10, 15, 20])
        >>> g200 = u.log_mag_err('g', gmags, 200)
        """
        _df = pd.read_csv(spline_csv)
        splines = dict()
        splines['g'] = self.__init_spline(_df, 'knots_G', 'coeff_G')
        splines['bp'] = self.__init_spline(_df, 'knots_BP', 'coeff_BP')
        splines['rp'] = self.__init_spline(_df, 'knots_RP', 'coeff_RP')
        self.__splines = splines
        self.__nobs_baseline = {'g': 200, 'bp': 20, 'rp': 20}
        self.n_obs = self.set_nobs(n_obs)

    def set_nobs(self, n_obs):
        """Set the number of observations for G, G_BP, G_RP bands.
        The numbers are proportional to the number of matched_transits in Gaia (can be computed via GaiaScanningLaw).
        """
        self.n_obs = None
        if isinstance(n_obs, int):
            if n_obs > 0:
                self.n_obs = {
                    'g': n_obs * 8.5, 'bp': n_obs * 0.95, 'rp': n_obs * 0.91
                }
        elif isinstance(n_obs, np.ndarray):
            self.n_obs = {}
            self.n_obs['g'] = n_obs * 8.5
            self.n_obs['bp'] = n_obs * 0.95
            self.n_obs['rp'] = n_obs * 0.91
        return

    def __init_spline(self, df, col_knots, col_coeff):
        __ddff = df[[col_knots, col_coeff]].dropna()
        return interpolate.BSpline(__ddff[col_knots], __ddff[col_coeff], 3, extrapolate=False)

    def log_mag_err(self, band, mag_val, n_obs=None):
        if n_obs is not None:
            self.set_nobs(n_obs)
        # If number of observations was not passed
        if self.n_obs is None:
            return 10 ** self.__splines[band](mag_val)

        return 10 ** (self.__splines[band](mag_val) - np.log10(
            np.sqrt(self.n_obs[band]) / np.sqrt(self.__nobs_baseline[band])))


class PhotHandler:
    def __init__(self, cluster_object, spline_csv):
        self.cluster_object = cluster_object
        print('Initializing the scanning law object...')
        self.sl = GaiaScanningLaw('dr3_nominal')
        self.n_obs = self.query_nobs()
        # Compute apparent magnitude uncertainties
        self.spline_csv = spline_csv
        self.u = None
        self.g_mag_err = None
        self.bp_mag_err = None
        self.rp_mag_err = None
        self.compute_uncertainties_phot()

    def compute_uncertainties_phot(self):
        self.u = Edr3LogMagUncertainty(self.spline_csv, self.n_obs)
        self.g_mag_err = self.u.log_mag_err('g', self.g_mag())
        self.bp_mag_err = self.u.log_mag_err('bp', self.bp_mag())
        self.rp_mag_err = self.u.log_mag_err('rp', self.rp_mag())

    def M_G(self):
        return self.cluster_object.data_phot['M_G'].values

    def M_Gbp(self):
        return self.cluster_object.data_phot['G_BP'].values

    def M_Grp(self):
        return self.cluster_object.data_phot['G_RP'].values

    def teff(self):
        return self.cluster_object.data_phot['teff'].values

    def logg(self):
        return self.cluster_object.data_phot['logg'].values

    def mass(self):
        return self.cluster_object.data_phot['mass'].values

    def lifetime(self):
        return self.cluster_object.data_phot['lifetime_logAge'].values

    def is_binary(self):
        return self.cluster_object.data_phot['is_binary'].values

    def apparent_mag(self, M):
        distance = self.cluster_object.skycoord.distance.value
        return M + 5 * np.log10(distance) - 5

    def g_mag(self):
        return self.apparent_mag(self.M_G())

    def bp_mag(self):
        return self.apparent_mag(self.M_Gbp())

    def rp_mag(self):
        return self.apparent_mag(self.M_Grp())

    def query_nobs(self):
        ra = self.cluster_object.skycoord.ra.value
        dec = self.cluster_object.skycoord.dec.value

        # Define helper function
        def get_totaln(*args):
            return sum(self.sl.query(*args, count_only=True))

        # Query the number of observations
        n_obs = [get_totaln(*args) for args in zip(ra, dec)]
        return np.array(n_obs)


class GaiaUncertainties(PhotHandler):
    def __init__(self, cluster_object: ClusterSampler, spline_csv: str, release: str = 'dr3'):
        """Takes effective temperature, surface gravity, absolute G, BP, RP magnitudes, and the distance.
        Computes parallax, radial velocity, and proper motion uncertainties.
        See: https://www.cosmos.esa.int/web/gaia/science-performance#spectroscopic%20performance
        """
        super().__init__(cluster_object, spline_csv)
        self.release = release
        # G_RVS maginitude
        self.g_rvs = self.compute_grvs(self.g_mag(), self.rp_mag())
        # Uncertainties to compute
        self.ra_err = None
        self.dec_err = None
        self.plx_err = None
        self.pmra_err = None
        self.pmdec_err = None
        self.rv_err = None
        self.completeness = None
        # Compute uncertainties
        self.compute_uncertainties_astrometry()
        # Estimate completeness
        print('Estimating completeness...')
        self.estimate_completeness()

    def new_cluster(self, cluster_object):
        self.cluster_object = cluster_object
        self.n_obs = self.query_nobs()
        self.compute_uncertainties_phot()
        self.compute_uncertainties_astrometry()
        self.estimate_completeness()
        return

    def estimate_completeness(self):
        """Estimate the completeness of the cluster"""
        # Compute the completeness
        mapHpx7 = DR3SelectionFunctionTCG()
        self.completeness = mapHpx7.query(self.cluster_object.skycoord, self.g_mag())
        # Completeness on the bright end
        c_bright = lambda x: 0.025 * x + 0.7
        cut = (self.g_mag() < 12) & (self.g_mag() >= 4)
        self.completeness[cut] = c_bright(self.g_mag()[cut])
        return

    def compute_parallax_uncertainty(self):
        """Compute parallax uncertainty in mas
        Function parallax_uncertainty returns paralax uncertainty in µas
        """
        if self.n_obs is None:
            self.plx_err = parallax_uncertainty(self.g_mag(), release=self.release) / 1_000.
        else:
            __nobs_baseline_plx = 200
            __nobs_astrometric_good = 8.3 * self.n_obs
            n_obs_relative = __nobs_astrometric_good / __nobs_baseline_plx
            self.plx_err = parallax_uncertainty(self.g_mag(), release=self.release) / 1_000. / np.sqrt(n_obs_relative)
        return

    def compute_uncertainties_astrometry(self):
        """Compute uncertainties for the astrometric parameters
        For the conversion factors see:
            https://www.cosmos.esa.int/web/gaia/science-performance#spectroscopic%20performance
        """
        self.compute_parallax_uncertainty()
        self.ra_err = 0.8 * self.plx_err
        self.dec_err = 0.7 * self.plx_err
        self.pmra_err = 1.03 * self.plx_err
        self.pmdec_err = 0.89 * self.plx_err
        self.rv_err = radial_velocity_uncertainty(self.g_mag(), self.teff(), self.logg(), release=self.release)
        return

    @staticmethod
    def compute_grvs(G, Grp):
        g_rp = G - Grp
        f1_g_rvs = -0.0397 - 0.2852 * g_rp - 0.033 * g_rp ** 2 - 0.0867 * g_rp ** 3
        f2_g_rvs = -4.0618 + 10.0187 * g_rp - 9.0532 * g_rp ** 2 + 2.6089 * g_rp ** 3
        # functions valid within the following ranges
        range_1 = g_rp < 1.2
        range_2 = 1.2 <= g_rp
        # Compute G_RVS
        if isinstance(G, np.ndarray):
            grvs = np.zeros_like(g_rp)
            grvs[range_1] = f1_g_rvs[range_1] + Grp[range_1]
            grvs[range_2] = f2_g_rvs[range_2] + Grp[range_2]
            return grvs
        else:
            if range_1:
                return f1_g_rvs + Grp
            else:
                return f2_g_rvs + Grp