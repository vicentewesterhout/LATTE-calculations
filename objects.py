import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from curves import *

class star:
    def __init__(self, name = 'Star', mag=9,temp='A0', filter_type='V',wl = ATM_WAV, ext = ATM_TRANS):
        self.name = name
        self.magnitude = mag
        self.filter_type = filter_type

        # check whether temp was given as a numeric value or as a spectral type string
        if isinstance(temp, str):
            self.T = teff_from_spectral_type(temp)
        else:
            self.T = temp

        self.flux = magnitude_2_flux(self.magnitude, type=self.filter_type)
        self.wl = wl
        self.ext = ext

class telescope:
    def __init__(self, name = 'Telescope', diameter = 2.54, obscuration = 0, wl = AL_BARE_WAV, refl = AL_BARE_REFL):
        self.name = name
        self.diameter = diameter
        self.obscuration = obscuration  

        # Effective collecting area of the telescope
        self.A = np.pi * (self.diameter / 2)**2 * (1 - self.obscuration)
        self.wl = wl
        self.refl = refl

class camera:
    def __init__(self, name = 'Camera', full_well=30000, bit_depth=16, wl = PCO_QE_WAV, ef = PCO_QE_EFF):
        self.name = name
        self.full_well_capacity = full_well      # Maximum electron capacity
        self.bit_depth = bit_depth                    # ADC bit depth
        self.max_adu = 2**self.bit_depth - 1  # Maximum ADU value for the given bit depth

        self.gain = self.full_well_capacity / self.max_adu  # Gain in e⁻/ADU
        self.wl = wl
        self.ef = ef
        

def magnitude_2_flux(mag, type='V',F0=3.62286e-11):
# def magnitud_a_flujo(mag, F0=2.81e-11):
    """
    Converts apparent magnitude (Vega system) to flux density in W/m²/nm.

    F0 value for lambda_centre = 550 nm, taken from the Vega magnitude-flux reference table.
    """
    if type == 'V':
        flujo = F0 * 10**(-0.4 * mag) # in W/m²/nm
        return flujo 
    elif type == 'G':
        ZP_G  = 25.6874        # zero point VEGAMAG Gaia DR3
        c_lambda = 1.346109e-21
        np_flux = 10**(0.4 * (ZP_G - mag))
        flujo = np_flux * c_lambda # in W/m²/nm
        return flujo

def planck_law(lam, T):
    """
    Planck's law for blackbody radiation.

    lam : wavelength in metres
    T   : temperature in Kelvin

    """
    h = 6.62607015e-34    # Planck's constant (J*s)
    c = 299792458         # Speed of light (m/s)
    kB = 1.380649e-23     # Boltzmann constant (J/K)


    return (2 * h * c**2 / lam**5) * (1 / (np.exp(h * c / (lam * kB * T)) - 1)) # in W/m²/m


def combine_efficiency_curves(
    wl1, eff1,
    wl2, eff2,
    wl3, eff3
):
    """
    Combines 3 efficiency curves by interpolating them onto a common wavelength grid.

    Parameters
    ----------
    wl1, eff1 : array-like  — (wavelength, efficiency) pair for curve 1
    wl2, eff2 : array-like  — (wavelength, efficiency) pair for curve 2
    wl3, eff3 : array-like  — (wavelength, efficiency) pair for curve 3

    Returns
    -------
    wl_ref : ndarray  — reference wavelength grid
    e1, e2, e3 : ndarray  — efficiencies interpolated onto wl_ref
    """
    arrays = {
        1: (np.asarray(wl1), np.asarray(eff1)),
        2: (np.asarray(wl2), np.asarray(eff2)),
        3: (np.asarray(wl3), np.asarray(eff3)),
    }

    lim_inf =[]
    lim_sup = []

    for key in ( 1, 2, 3):
        wl_ref,_ = arrays[key]
        lim_inf.append(wl_ref[0])
        lim_sup.append(wl_ref[-1])


    ref = np.arange(max(lim_inf), min(lim_sup),2)  # Common grid between the overlapping wavelength limits

    result = []
    for key in (1, 2, 3):
        wl_i, eff_i = arrays[key]
        interp = interp1d(
            wl_i, eff_i,
            kind='cubic',
            bounds_error=False,
            fill_value=0.0
        )
        result.append(interp(ref))

    return ref, result[0], result[1], result[2]

def ph_count_integral(star,wl,qe,atm,refl,telescope,exp_time, wl_v=lambda_V, trans_v=trans_V, verbose=False):
        
    photon_flux_outside, photon_flux_ground, photon_flux_telescope, photon_flux_detected = flux_curves(star, wl,qe,atm,refl,wl_v=wl_v, trans_v=trans_v)
    photon_count_outside = np.trapezoid(photon_flux_outside, wl)
    photon_count_ground  = np.trapezoid(photon_flux_ground, wl)
    photon_count_telescope = np.trapezoid(photon_flux_telescope, wl) * telescope.A  
    photon_count_detected = np.trapezoid(photon_flux_detected, wl) * telescope.A * exp_time 

    if verbose:
        # Plotting the flux curves
        plt.figure(figsize=(10, 6))
        plt.plot(wl, photon_flux_outside, label='Top of Atmosphere', color='blue')
        plt.plot(wl, photon_flux_ground, label='Ground level', color='green')
        plt.plot(wl, photon_flux_telescope, label='Telescope', color='red')
        plt.plot(wl, photon_flux_detected, label='Detected', color='purple')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux Density (photons/s/m²/nm)')
        plt.title(f'Flux Curves for {star.name}, Mag {star.magnitude}, {"Temp" if isinstance(star.T, (int, float)) else "Type"}={star.T}{" K" if isinstance(star.T, (int, float)) else ""}')
        plt.legend()
        plt.show()
    
    return photon_count_outside, photon_count_ground, photon_count_telescope,photon_count_detected


def flux_curves(star, wl, qe, atm, refl, wl_v=lambda_V, trans_v=trans_V):
    # Computes photon flux curves for a given star, accounting for camera quantum efficiency, atmospheric extinction, and telescope reflectivity.

    h = 6.62607015e-34    # Planck's constant (J*s)
    c = 299792458         # Speed of light (m/s)
    f = h*c/(wl*1e-9)  # energy of a single photon in J

    c = (np.trapezoid(planck_law(wl_v*1e-9, star.T)*trans_v, wl_v) /
         np.trapezoid(trans_v, wl_v))  # V-filter weighted mean → scales the Planck model to W/m²/nm at λ_eff

    C = star.flux/c
    flux_outside = planck_law(wl*1e-9, star.T) * C # flux in W/m²/nm at the wavelength of interest

    photon_flux_outside = flux_outside/f # photon flux in photons/m²/s
    photon_flux_ground  = photon_flux_outside * atm                                # photon flux after atmospheric extinction
    photon_flux_telescope = photon_flux_ground * refl**2                           # photon flux after telescope reflectivity
    photon_flux_detected = photon_flux_telescope * qe                              # photon flux detected by the camera in e⁻/m²/s

    return photon_flux_outside, photon_flux_ground, photon_flux_telescope, photon_flux_detected


def teff_from_spectral_type(sp_type: str) -> float:
    """
    Estimates Teff from spectral type (e.g. 'G2', 'K5', 'A0').
    If the exact subtype is not in the table, interpolates between the nearest entries.
    """
    sp_type = sp_type.strip().upper()[:2]  # keep only letter + digit, e.g. 'G2'
    
    if sp_type in SPECTRAL_TEFF:
        return SPECTRAL_TEFF[sp_type]
    
    # Interpolation: find nearest neighbours
    letter = sp_type[0]
    number = int(sp_type[1])
    
    candidates = {k: v for k, v in SPECTRAL_TEFF.items() if k[0] == letter}
    keys_sorted = sorted(candidates.keys(), key=lambda x: int(x[1]))
    
    for i in range(len(keys_sorted) - 1):
        n1 = int(keys_sorted[i][1])
        n2 = int(keys_sorted[i+1][1])
        if n1 <= number <= n2:
            t1 = candidates[keys_sorted[i]]
            t2 = candidates[keys_sorted[i+1]]
            t = t1 + (t2 - t1) * (number - n1) / (n2 - n1)
            return round(t)
    
    raise ValueError(f"Spectral type '{sp_type}' is out of range of the built-in table.")


def compute_observation_flux(star,telescope,camera,exp_time=0.015,verbose = True, wl_v=lambda_V, trans_v=trans_V):

    # efficiency curves
    wl_qe = camera.wl
    ef_qe = camera.ef
    wl,qe,atm,refl = combine_efficiency_curves(
        camera.wl, camera.ef,
        star.wl, star.ext,
        telescope.wl, telescope.refl)

    photon_count_outside, photon_count_ground, photon_count_telescope,photon_count_detected = ph_count_integral(star,wl,qe,atm,refl,telescope,exp_time, wl_v=wl_v, trans_v=trans_v,verbose=verbose)
    ADU_count = photon_count_detected/camera.gain

    if verbose:

        # Plot efficiency curves vs wavelength 
        plt.figure(figsize=(10, 6))
        plt.plot(wl, qe, label='QE PCO Camera', color='blue')
        # plt.plot(wl, atm, label=f'Atmospheric Extinction for zenith angle {zen_angle}°', color='green')
        plt.plot(wl, atm, label=f'Atmospheric Extinction', color='green')
        plt.plot(wl, refl**2, label='Enhanced Aluminum Reflectivity', color='red')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Efficiency [%]')
        plt.title('Efficiency Curves vs Wavelength')
        plt.legend()
        plt.show()

        print(f"Photon count outside atmosphere for {star.name}: {photon_count_outside:.2e} photons")
        print(f"Photon count at ground level for {star.name}: {photon_count_ground:.2e} photons")
        print(f"Photon count after telescope for {star.name}: {photon_count_telescope:.2e} photons")
        print(f"Photon count detected for {star.name} with camera {camera.name}: {photon_count_detected:.2e} photons")
        print(f"ADU count for {star.name} with {telescope.name} and {camera.name}: {ADU_count:.2f} ADU")

    return photon_count_outside, photon_count_ground, photon_count_telescope,photon_count_detected,ADU_count
