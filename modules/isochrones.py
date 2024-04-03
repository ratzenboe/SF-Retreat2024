import re
import os
import glob
import copy
import imf
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

# ----- Corrective factors for extinction correction -----
corr_Gmag = 0.83627
corr_BPmag = 1.08337
corr_RPmag = 0.63439
corr_bprp = corr_BPmag - corr_RPmag
corr_grp = corr_Gmag - corr_RPmag
# --------------------------------------------------------


class ICBase:
    def __init__(self):
        self.data = None
        self.colnames = None
        self.l_interp = None

    def apply_extinction_by_color(self, abs_mag_g, mag_bp, mag_rp, a_v):
        """Apply extinction to individual isochrone"""
        if isinstance(a_v, np.ndarray):
            av_add = a_v[None, :]
        elif isinstance(a_v, (list, tuple)):
            av_add = np.array(a_v)[None, :]
        else:
            av_add = a_v
        # Compute extincted magnitudes and colors
        extincted_mag_g = abs_mag_g + av_add * corr_Gmag
        extincted_mag_bp = mag_bp + av_add * corr_BPmag
        extincted_mag_rp = mag_rp + av_add * corr_RPmag
        return extincted_mag_g, extincted_mag_bp, extincted_mag_rp

    def fit_interpolator(self, n_skip=10):
        df_subset = self.data[::n_skip]
        cols_input = [self.colnames['mass'], self.colnames['age'], self.colnames['metal']]
        cols_predict = [
            self.colnames['gmag'], self.colnames['bp'], self.colnames['rp'],
            self.colnames['logg'], self.colnames['teff']
        ]
        X = df_subset[cols_input].values
        y = df_subset[cols_predict].values
        self.l_interp = LinearNDInterpolator(X, y)

    def query_cmd(self, mass, age, metal, av=0):
        if not isinstance(mass, np.ndarray):
            mass = np.array([mass])
            age = np.array([age])
            metal = np.array([metal])
        # Query the interpolator
        X_query = np.vstack([mass, age, metal]).T
        # Interpolate
        abs_mag_g, abs_mag_bp, abs_mag_rp, logg, teff = self.l_interp(X_query).T
        # Apply extinction
        # abs_mag_g_ext, mag_bp_ext, mag_rp_ext = self.apply_extinction_by_color(
        #     abs_mag_g, mag_bp, mag_rp, av
        # )
        df = pd.DataFrame(
            np.vstack([abs_mag_g, abs_mag_bp, abs_mag_rp, logg, teff]).T,
            columns=['M_G', 'G_BP', 'G_RP', 'logg', 'teff']
        )
        return df


class PARSEC(ICBase):
    """Handling Gaia (E)DR3 photometric system"""

    def __init__(self, dir_path, file_ending='dat'):  # , nb_interpolated=400):
        super().__init__()
        # Save some PARSEC internal column names
        self.comment = r'#'
        self.colnames = {
            'mass': 'Mass',
            'logg': 'logg',
            'teff': 'logTe',
            'age': 'logAge',
            'metal': 'MH',
            'gmag': 'Gmag',
            'bp': 'G_BPmag',
            'rp': 'G_RPmag',
            'header_start': '# Zini'
        }
        self.post_process = {self.colnames['teff']: lambda x: 10 ** x}
        # Save data and rename columns
        self.dir_path = dir_path
        self.flist_all = glob.glob(os.path.join(dir_path, f'*.{file_ending}'))
        self.data = self.read_files(self.flist_all)
        # Prepare interpolation method
        self.fit_interpolator(n_skip=5)

    def read_files(self, flist):
        frames = []
        for fname in flist:
            df_iso = self.read(fname)
            # Postprocessing
            for col, func in self.post_process.items():
                df_iso[col] = df_iso[col].apply(func)
            frames.append(df_iso)
        print('PARSEC isochrones read and processed!')
        return pd.concat(frames)

    def read(self, fname):
        """
        Fetches the coordinates of a single isochrone from a given file and retrurns it
        :param fname: File name containing information on a single isochrone
        :return: x-y-Coordinates of a chosen isochrone, age, metallicity
        """
        df_iso = pd.read_csv(fname, delim_whitespace=True, comment=self.comment, header=None)
        # Read first line and extract column names
        with open(fname) as f:
            for line in f:
                if line.startswith(self.colnames['header_start']):
                    break
        line = re.sub(r'\t', ' ', line)  # remove tabs
        line = re.sub(r'\n', ' ', line)  # remove newline at the end
        line = re.sub(self.comment, ' ', line)  # remove '#' at the beginning
        col_names = [elem for elem in line.split(' ') if elem != '']
        # Add column names
        df_iso.columns = col_names
        return df_iso


class Baraffe15(ICBase):
    """Handling Gaia (E)DR3 photometric system"""

    def __init__(self, dir_path, file_ending='GAIA'):
        super().__init__()
        # Save some PARSEC internal column names
        self.comment = r'!'
        self.colnames = {
            'mass': 'M/Ms',
            'logg': 'g',
            'teff': 'Teff',
            'age': 'logAge',
            'metal': 'feh',
            'gmag': 'G',
            'bp': 'G_BP',
            'rp': 'G_RP',
            'header_start': '! M/Ms',
            'age_start': '!  t (Gyr) ='
        }
        # Save data and rename columns
        self.dir_path = dir_path
        self.flist_all = glob.glob(os.path.join(dir_path, f'*.{file_ending}'))
        self.data = self.read_files(self.flist_all)
        # Prepare interpolation method
        self.fit_interpolator(n_skip=1)

    def read_files(self, flist):
        frames = []
        for fname in flist:
            df_iso = self.read(fname)
            frames.append(df_iso)
        print('Baraffe+15 isochrones read and processed!')
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames)

    def read(self, fname):
        """
        Fetches the coordinates of a single isochrone from a given file and retrurns it
        :param fname: File name containing information on a single isochrone
        :return: x-y-Coordinates of a chosen isochrone, age, metallicity
        """
        df_iso = pd.read_csv(fname, delim_whitespace=True, header=None, comment=self.comment)
        # Read first line and extract column names
        with open(fname) as f:
            for line in f:
                if line.startswith(self.colnames['header_start']):
                    break
        line = re.sub(r'\t', ' ', line)  # remove tabs
        line = re.sub(r'\n', ' ', line)  # remove newline at the end
        line = re.sub(self.comment, ' ', line)  # remove '!' at the beginning
        col_names = [elem for elem in line.split(' ') if elem != '']
        # Add column names
        df_iso.columns = col_names
        # Post process: add ages
        counter = 0
        nb_entries_per_isochrone = []
        logAge_info = []
        with open(fname) as f:
            line_info = []
            for i, line in enumerate(f):
                if line.startswith('!---'):
                    counter += 1
                    if counter == 2:
                        line_info.append(i + 1)
                    if counter == 3:
                        line_info.append(i - 1)

                if line.startswith(self.colnames['age_start']):
                    line = re.sub(r'\t', ' ', line)  # remove tabs
                    age = float(re.findall("\d+\.\d+", line)[0])
                    logAge = np.round(np.log10(age * 10 ** 9), decimals=2)
                    logAge_info.append(logAge)
                    # Save infos
                    if len(line_info) > 0:
                        counter = 0
                        nb_entries_per_isochrone.append(line_info[1] - line_info[0])
                        line_info = []
        nb_entries_per_isochrone.append(line_info[1] - line_info[0])
        # --- Add age ---
        df_iso[self.colnames['age']] = -1.0
        rolling_sum = 0
        for entry, logAge in zip(nb_entries_per_isochrone, logAge_info):
            end = entry + rolling_sum + 1
            df_iso[self.colnames['age']].iloc[rolling_sum:end] = logAge
            rolling_sum = end
        # --- Add metal (at least 2 points to allow interpolation) ---
        # todo: fix interpolator to a single argument
        df_iso[self.colnames['metal']] = -1.
        df_iso_1 = copy.deepcopy(df_iso)
        df_iso_1[self.colnames['metal']] = 1.
        data = pd.concat([df_iso, df_iso_1])
        return data


class ClusterPhotometry:
    def __init__(self, parsec_folder, baraffe_folder, M_G_threshold=5):
        self.p_obj = PARSEC(parsec_folder)
        self.b_obj = Baraffe15(baraffe_folder)
        self.M_G_threshold = M_G_threshold
        self.source_data_photometry = None
        self.data_photometry_binaries = None
        self.data_phot = None

    def make_photometry(self, cluster_mass, logAge, feh=0, f_bin_total=0.5):
        mass_samples = imf.make_cluster(cluster_mass)
        mass_samples = np.sort(mass_samples)
        logAge_samples = np.full_like(mass_samples, logAge)
        feh_samples = np.full_like(mass_samples, feh)

        df_parsec = self.p_obj.query_cmd(mass_samples, logAge_samples, feh_samples)
        df_baraffe = self.b_obj.query_cmd(mass_samples, logAge_samples, feh_samples)
        df_parsec['mass'] = mass_samples
        df_baraffe['mass'] = mass_samples

        idx_parsec = np.where(df_parsec.M_G < self.M_G_threshold)[0].min()
        # Use Baraffe for lower main sequence, PARSEC for upper MS
        self.source_data_photometry = pd.concat([df_baraffe[:idx_parsec], df_parsec[idx_parsec:]])
        if f_bin_total > 0:
            self.add_binaries(f_bin_total)
            self.data_phot = self.data_photometry_binaries
        else:
            self.data_phot = self.source_data_photometry
        self.compute_lifetimes()
        return self.data_phot

    @staticmethod
    def add_magnitudes(*args):
        return -2.5 * np.log10(np.sum([10 ** (-0.4 * M_i) for M_i in args], axis=0))

    def create_binaries_pairs(self, f_bin_total):
        n = self.source_data_photometry.shape[0]
        random_idx_pairs = np.array([[i, j] for i, j in zip(np.arange(n), np.random.permutation(n))])
        rand_pairs_boolarr = np.random.uniform(0, 1, n) < f_bin_total
        random_idx_pairs_filtered = random_idx_pairs[rand_pairs_boolarr]
        # Sort by joint mass
        random_idx_pairs_filtered = random_idx_pairs_filtered[random_idx_pairs_filtered.sum(axis=1).argsort()][::-1]
        all_pairs_final = []
        unique_sources = set()
        for i, j in random_idx_pairs_filtered:
            if (i not in unique_sources) and (j not in unique_sources):
                all_pairs_final.append([i, j])
                unique_sources.add(i)
                unique_sources.add(j)
        return np.array(all_pairs_final)

    def add_binaries(self, f_bin_total):
        all_pairs_final = self.create_binaries_pairs(f_bin_total)
        max_nb = np.max(all_pairs_final, axis=1)
        min_nb = np.min(all_pairs_final, axis=1)
        # Create copy to store binaries in
        self.data_photometry_binaries = self.source_data_photometry.copy()
        # Get photometry of given ids
        df_max = self.data_photometry_binaries.loc[max_nb]
        df_min = self.data_photometry_binaries.loc[min_nb]
        # Compute the combined photometry
        for col in ['M_G', 'G_BP', 'G_RP']:
            self.data_photometry_binaries.loc[max_nb, col] = self.add_magnitudes(
                df_max[col].values, df_min[col].values
            )
        # Compute the combined logg and teff
        for col in ['logg', 'teff']:
            self.data_photometry_binaries.loc[max_nb, col] = np.max(
                np.vstack([df_max[col].values, df_min[col].values]), axis=0
            )
        # Compute combined mass
        self.data_photometry_binaries.loc[max_nb, 'mass'] = df_max['mass'].values  # + df_min['mass'].values
        # Save is binary as columns
        self.data_photometry_binaries['is_binary'] = 0.
        self.data_photometry_binaries.loc[max_nb, 'is_binary'] = 1.
        self.data_photometry_binaries.loc[min_nb, 'is_binary'] = -1.
        # Remove min_nb from the source data
        self.data_photometry_binaries = self.data_photometry_binaries.drop(min_nb)
        return

    def compute_lifetimes(self):
        mass = self.data_phot['mass'].values
        self.data_phot['lifetime_logAge'] = np.log10(10**10 * (1/mass)**2.5)
