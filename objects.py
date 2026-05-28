import numpy as np
import astropy.units as u
import astropy.constants as const


class star:
    def __init__(self, name, magnitude, lam_centre = 550):
        self.name = name
        self.magnitude = magnitude
        self.lam_centre = lam_centre * 1e-9 # en metros

        self.flux_hz = magnitud_a_flujo(self.magnitude)  # en W/m²/Hz
        self.flux_lam = (self.flux_hz * const.c.value / (self.lam_centre)**2) # en W/m**3 

class telescope:
    def __init__(self, name, diameter, obscuration, efficiency, focal_length):
        self.name = name
        self.diameter = diameter
        self.obscuration = obscuration
        self.efficiency = efficiency
        self.focal_length = focal_length

        # Área efectiva del telescopio
        self.A = np.pi * (self.diameter / 2)**2 * (1 - self.obscuration)

class camera:
    def __init__(self, name, quantum_efficiency, bandwidth, read_noise, dark_current, full_well, bit_depth,pixel_size,width,height):
        self.name = name
        self.qe = quantum_efficiency         #Quantum efficiency
        self.bandwidth = bandwidth * 1e-9    #Ancho de banda en [m]
        self.read_noise = read_noise               #Ruido de lectura en e⁻
        self.dark_current = dark_current  #Corriente de oscuridad en e
        self.full_well_capacity = full_well      #Capacidad máxima de electrones
        self.max_adu = 2**bit_depth - 1  # Máximo valor de ADU para el bit depth dado
        self.pixel_size = pixel_size * 1e-6 # Tamaño del píxel en metros

        self.gain = self.full_well_capacity / self.max_adu  # Ganancia en e⁻/ADU
        self.width = width
        self.height = height


# Funciones auxiliares

def magnitud_a_flujo(mag, F0=3631 ):
    """
    Convierte magnitud aparente (sistema AB) a flujo por unidad de frecuencia en Jy.
    """
    flujo = F0 * 10**(-0.4 * mag) # en Jy = e-26 W/m²/Hz
    return flujo*1e-26 # Convertir a W/m²/Hz 


class system_propagation:
    def __init__(self, star, telescope, camera,t_exp):
        self.star = star
        self.telescope = telescope
        self.camera = camera
        self.t_exp  = t_exp         # tiempo de exposición

        # Energía por fotón en la longitud de onda central
        self.E_foton = (const.h.value * const.c.value / self.star.lam_centre) # en [J]
        # Flujo total integrado en la banda 
        self.flujo_total = (self.star.flux_lam * self.camera.bandwidth) # en [W/m²]
        # Fotones detectados
        self.N_fotones = ((self.flujo_total * self.telescope.A * self.camera.qe * self.t_exp) / self.E_foton)
        # Plate scale
        self.plate_scale = (206265 * self.camera.pixel_size / self.telescope.focal_length)
        # FWHM por difracción
        self.FWHM_diff = 1.22 * 206265 * (self.star.lam_centre / self.telescope.diameter )
        self.FWHM_px    = (self.FWHM_diff / self.plate_scale)
        self.sigma_px = self.FWHM_px / (2 * np.sqrt(2 * np.log(2)))
        self.Psf_peak = self.N_fotones / (2* np.pi * self.sigma_px**2)
