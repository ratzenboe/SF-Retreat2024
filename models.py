# ----- Imports ------

import nifty8.re as jft
import jax.numpy as np
import pandas as pd
from itertools import chain


class SigmoidField(jft.Model):
    def __init__(self, amp_mean: float , amp_std: float, cf: jft.Model, name: str):
        self.cf = cf
        self.amp = jft.NormalPrior(amp_mean, amp_std, name=name)
        super().__init__(init=cf.init | self.amp.init)
    
    def __call__(self, x):
        return self.amp(x)/(1. + np.exp(-self.cf(x)))
       


class logStellarAges(jft.Model):
    def  __init__(self, bg_mean_field: jft.Model, c_mean_field: jft.Model, bg_std_field: jft.Model, c_std_field: jft.Model, n_data: int, coordinates : np.ndarray):
        self.bg_mean = bg_mean_field
        self.bg_std = bg_std_field
        self.c_mean = c_mean_field
        self.c_std = c_std_field
        
        self.excitations = jft.NormalPrior(mean=0., std=1., shape=(n_data,), dtype=float, name="xi_age")
        self.coordinates = np.floor(coordinates).astype(int)
        super().__init__(init=bg_mean_field.init | c_mean_field.init | bg_std_field.init | c_std_field.init | self.excitations.init)
    
    def __call__(self, x):
        mean = self.bg_mean(x)[self.coordinates] + self.c_mean(x)[self.coordinates]
        std = self.bg_std(x)[self.coordinates] + self.c_std(x)[self.coordinates]
        return mean + std*self.excitations(x)
    

class StarsToLuminosity(jft.Model):
    def __init__(self, mass_model: jft.Model, age_model: jft.Model,  constant_metal: bool = True, feh : int = 0):
        self.constant_metal = constant_metal
        self.mass_model = mass_model
        self.age_model = age_model
        self.feh = feh
        super.__init__(init=mass_model.init | age_model.init)
        
    def __call__(self, x):
        
        mass = self.mass_model(x)
        logAge = self.age_model(x)
        
        df_parsec = self.p_obj.query_cmd(mass, logAge, self.feh)
        #
        # df_baraffe = self.b_obj.query_cmd(mass, logAge, feh)
        df_parsec['mass'] = mass
        #df_baraffe['mass'] = mass
        
        # idx_parsec = np.where(df_parsec.M_G < self.M_G_threshold)[0].min()
        # Use Baraffe for lower main sequence, PARSEC for upper MS
        return df_parsec

