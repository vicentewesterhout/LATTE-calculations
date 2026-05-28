import numpy as np
import astropy.units as u
import astropy.constants as const

from objects import*

import photutils.datasets as phu
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.visualization import ZScaleInterval

from photutils.psf import CircularGaussianPSF, GaussianPRF, GaussianPSF


star1 = star("HD180530", 8.9)
star2 = star("HIP101974", 6.9)
INT   = telescope("INT", 2.54, 0, 1, 4.469)
CEL = telescope("Celestron 14 EdgeHD", 0.3556, 0, 1, 0.79)
pco_edge_55 = camera("pco_edge_5.5", 0.60, 400,1,0.6,30000,16,6.5,51,51) 

# Parámetros generales
t_exp  = 0.015         # tiempo de exposición


system1 = system_propagation(star1, INT, pco_edge_55,t_exp)
system2 = system_propagation(star2, INT, pco_edge_55,t_exp)

# # Energía por fotón en la longitud de onda central
# E_foton = (const.h.value * const.c.value / star1.lam_centre) # en [J]
# print(f"Photon energy: {system1.E_foton:.3e} J")

# # Flujo total integrado en la banda 
# flujo_total = (star1.flux_lam * pco_edge_55.bandwidth) # en [W/m²]
# print(f"Spectral flux density in the V-band: {system1.flujo_total:.3e} W/m²")

# # Fotones detectados
# N_fotones = (flujo_total * INT.A * pco_edge_55.qe * t_exp / E_foton)
# print(f"\n⭐ Photon count: {system1.N_fotones:.4e}")

# # Plate scale
# plate_scale = (206265 * pco_edge_55.pixel_size / INT.focal_length)
# print(f"\n📐 Plate scale: {system2.plate_scale:.3f} arcsec/px ")

# # FWHM por difracción

# FWHM_diff = 1.22 * 206265 * (star1.lam_centre / INT.diameter )
# print(f"\n🔭 FWHM difracción: {system1.FWHM_diff:.3f} arcsec")

# FWHM_px    = (FWHM_diff / plate_scale)
# print(f"\n📏 FWHM: {system1.FWHM_px:.2f} px")

# sigma_px = FWHM_px / (2 * np.sqrt(2 * np.log(2)))
# print(f"📏 Sigma:    {system1.sigma_px:.4f} px") 

# A = N_fotones / (2* np.pi * sigma_px**2)
# print(f"\n🔢 Amplitud de la PSF: {A:.3e} e⁻/px")


# -----------------------------------------------
# Simulación de imagen con escala física correcta
# -----------------------------------------------

seeing = 3.0  # arcsec — seeing atmosférico (domina sobre difracción)

# FWHM efectiva: cuadratura de difracción + seeing, convertida a píxeles
fwhm1_px = np.sqrt(system1.FWHM_diff**2 + seeing**2) / system1.plate_scale
fwhm2_px = np.sqrt(system2.FWHM_diff**2 + seeing**2) / system2.plate_scale

print(f"\n⭐ {star1.name}  N_fotones={system1.N_fotones:.3e}  FWHM_eff={fwhm1_px:.2f} px")
print(f"⭐ {star2.name}  N_fotones={system2.N_fotones:.3e}  FWHM_eff={fwhm2_px:.2f} px")

model1 = CircularGaussianPSF(flux=system1.N_fotones, x_0=pco_edge_55.width//2,      y_0=pco_edge_55.height//2, fwhm=fwhm1_px)
model2 = CircularGaussianPSF(flux=system2.N_fotones, x_0=pco_edge_55.width//2 + 20, y_0=pco_edge_55.height//2, fwhm=fwhm2_px)
yy, xx = np.mgrid[0:pco_edge_55.height, 0:pco_edge_55.width]

data1 = model1(xx, yy)
data2 = model2(xx, yy)
data  = data1 + data2

data_adu = data / pco_edge_55.gain  # Convertir a ADU (imagen ideal, sin ruido)

# Imagen realista: ruido de Poisson (aprox. Gaussiana para lambda grande) + oscuridad + lectura
safe_data   = np.maximum(data, 0)
data_shot   = np.random.normal(safe_data, np.sqrt(safe_data))          # shot noise (Poisson ~ N(λ,√λ))
data_shot  += np.random.poisson(pco_edge_55.dark_current * t_exp, size=data.shape).astype(float)  # dark current
data_shot  += np.random.normal(0, pco_edge_55.read_noise, size=data.shape)                        # read noise
bias = 500  # ADU — pedestal electrónico (como en detectores reales)
data_noisy_adu = data_shot / pco_edge_55.gain + bias
data_quantized = np.clip(data_noisy_adu, 0, pco_edge_55.max_adu).astype(int)  # Cuantizar y limitar al máximo ADU

# Límites LogNorm seguros por imagen (evita vmin >= vmax)
# def safe_lognorm(arr):
#     pos = arr[arr > 0]
#     if len(pos) == 0 or arr.max() <= 0:
#         return LogNorm(vmin=1, vmax=10)
#     vmax = float(arr.max())
#     vmin = float(max(pos.min(), vmax * 1e-6))
#     if vmin >= vmax:
#         vmin = vmax / 10
#     return LogNorm(vmin=vmin, vmax=vmax)

# # Crear una figura con las 3 imagenes en una fila
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# # Imagen original
# axes[0].imshow(data, origin='lower', cmap='viridis',norm =LogNorm())
# axes[0].set_title('Photon count (e⁻/px)')
# # Imagen en ADU
# axes[1].imshow(data_adu, origin='lower', cmap='viridis', norm = LogNorm())
# axes[1].set_title('ADU Image')
# # Imagen cuantizada (con ruido realista)
# axes[2].imshow(data_quantized, origin='lower', cmap='viridis', vmin=0, vmax=pco_edge_55.max_adu)
# axes[2].set_title('Quantized + Noise (ADU)')
# for ax in axes:
#     ax.set_xlabel('X (px)')
#     ax.set_ylabel('Y (px)')
# plt.tight_layout()
# plt.show()



##########################################################################################

# Simulacion para laser de Na en la capa de Sodio

##########################################################################################

H_na = 90000 # Altura de la capa de sodio en metros
lam_na = 589.0 * 1e-9 # Longitud de onda del laser de metros
F_lgs = 90000 # Foco del laser en metros
w_pupil = 0.3 # Diámetro de la pupila de salida del laser en metros
b = 40.0 # Distancia entre el lanzador y la subapertura del telescopio en metros

dNA = 10000 # Grosor de la capa de sodio en metros
alt_lgs = 60 # Altitud del telescopio del laser de salida en metros
azi_lgs = 45 # Azimut del laser con respecto al telescopio en grados

## Cálculo del tamaño angular del penacho LGS
hip = (H_na +(dNA/2))/np.sin(np.radians(alt_lgs)) 
l_lgs = (hip * (H_na - (dNA/2))) / H_na
b_prime = hip * np.cos(np.radians(alt_lgs))
TO = np.sqrt(b_prime**2 + b**2 - 2*b*b_prime*np.cos(np.radians(azi_lgs)))
l_lgs_prime = b_prime - ((b_prime * (H_na - (dNA/2))) / H_na)
TO_prime = np.sqrt((b_prime - l_lgs_prime)**2 + b**2 - 2*b*(b_prime - l_lgs_prime)*np.cos(np.radians(azi_lgs)))
TL = np.sqrt(TO**2 +(H_na + (dNA/2))**2)
TL_prime = np.sqrt(TO_prime**2 + (H_na - (dNA/2))**2)
alfa = np.arccos((b**2 + TL**2 -hip**2)/(2*b*TL))
alfa_prime = np.arccos((b**2 + TL_prime**2 - (hip-l_lgs)**2)/(2*b*TL_prime))
beta = alfa - alfa_prime # Tamaño angular del penacho LGS en radianes
beta_arcsec = beta * (180/np.pi) * 3600 # Convertir a arcsec
# print(f"Angular size of the LGS plume: {beta_arcsec:.3f} arcsec") 


w_0 = (lam_na * F_lgs) / (np.pi * w_pupil/2) # Radio de waists del laser en metros
w_0 = 0.3
# print(f"Waist radius (w_0): {w_0:.3e} m")

# Tamaño angular de la cintura en radianes
theta = w_0 / F_lgs # en radianes
# Convertir a arcsec
theta_arcsec = theta * (180/np.pi) * 3600   
# print(f"Angular size of the waist: {theta_arcsec:.3f} arcsec")

# FWHM angular de la cintura
FWHM_theta = 2 * np.sqrt(2 * np.log(2)) * theta # en radianes
FWHM_theta_arcsec = FWHM_theta * (180/np.pi) * 3600
FWHM_theta_px = FWHM_theta_arcsec / system2.plate_scale
sigma_x = FWHM_theta_arcsec / (2 * np.sqrt(2 * np.log(2))) # Sigma de la PSF en arcsec
# print(f"FWHM angular size of the waist: {FWHM_theta_arcsec:.3f} arcsec")
# print(f"Sigma x of the PSF in arcsec: {sigma_x:.3f} arcsec")

#tamaño que ocuparía el penacho en la cámara en la otra dirección (en px)
plume_size_arcsec = 138
plume_size_px = plume_size_arcsec / 0.3

#Suponemmos que el laser tiene una magnitud aparente de 6.6 en la banda V, lo que corresponde a un flujo de:

flux_lgs = magnitud_a_flujo(6.6)
flux_lgs_lam = (flux_lgs * const.c.value / (550*1e-9)**2) # en W/m**3

N_fotones_lgs = (flux_lgs_lam * INT.A * pco_edge_55.qe * t_exp / (const.h.value * const.c.value / (550*1e-9)))
print(f"Photon count from LGS: {N_fotones_lgs:.3e} photons")

# sigma_y tal que la extensión visible sobre el ruido sea exactamente plume_size_px.
# La señal cae al nivel de ruido en: y = sigma_y * sqrt(2 * ln(A_peak / noise_floor))
# donde A_peak = N / (2*pi*sigma_x*sigma_y)  →  solución iterativa
noise_floor_e = np.sqrt(pco_edge_55.read_noise**2 + pco_edge_55.dark_current * t_exp)
sigma_x_px    = FWHM_theta_px / (2 * np.sqrt(2 * np.log(2)))
target_half   = plume_size_px / 2  # semi-extensión objetivo [px]

sigma_y = plume_size_px / 8  # estimación inicial
# for _ in range(50):
#     A_peak  = N_fotones_lgs / (2 * np.pi * sigma_x_px * sigma_y)
#     sigma_y = target_half / np.sqrt(2 * np.log(A_peak / noise_floor_e))
# fwhm_y = sigma_y * 2 * np.sqrt(2 * np.log(2))
# print(f"sigma_y: {sigma_y:.2f} px  →  fwhm_y: {fwhm_y:.2f} px")
sigma_y = 31.83
fwhm_y = sigma_y * 2 * np.sqrt(2 * np.log(2))
print(f"sigma_y: {sigma_y:.2f} px  →  fwhm_y: {fwhm_y:.2f} px")

# -----------------------------------------------
# Simulación de imagen del penacho LGS con perfil
# -----------------------------------------------

data3 = GaussianPRF(flux=N_fotones_lgs, x_0=pco_edge_55.width/2, y_0=pco_edge_55.height/2, x_fwhm=FWHM_theta_px, y_fwhm=fwhm_y)
# data3 = GaussianPSF(flux=1000, x_0=pco_edge_55.width/2, y_0=pco_edge_55.height/2, x_fwhm=5, y_fwhm=10)
yy, xx = np.mgrid[0:pco_edge_55.height, 0:pco_edge_55.width]
data_lgs = data3(xx, yy)
#sumar los pixeles para obtener el total de fotones
total_fotones_lgs = data_lgs.sum()
print(f"Total photons in the LGS image: {total_fotones_lgs:.3e} photons")
yy2, xx2 = np.mgrid[0:5000, 0:5000]
data_lgs_large = data3(xx2, yy2)
total_fotones_lgs_large = data_lgs_large.sum()
print(f"Total photons in the large LGS image: {total_fotones_lgs_large:.3e} photons")



lgs_peak = max(data_lgs.max(), data_lgs_large.max())
vmin_lgs = max(lgs_peak * 1e-6, 1e-3)

data_lgs_adu = data_lgs / pco_edge_55.gain  # Convertir a ADU (imagen ideal, sin ruido)
data_lgs_large_adu = data_lgs_large / pco_edge_55.gain  # Convertir a ADU (imagen ideal, sin ruido)
# Imagen realista: ruido de Poisson (aprox. Gaussiana para lambda grande) + oscuridad + lectura
safe_data_lgs   = np.maximum(data_lgs, 0)
data_lgs_shot   = np.random.normal(safe_data_lgs, np.sqrt(safe_data_lgs))          # shot noise (Poisson ~ N(λ,√λ))
data_lgs_shot  += np.random.poisson(pco_edge_55.dark_current * t_exp, size=data_lgs.shape).astype(float)  # dark current
data_lgs_shot  += np.random.normal(0, pco_edge_55.read_noise, size=data_lgs.shape)                        # read noise
data_lgs_noisy_adu = data_lgs_shot / pco_edge_55.gain + bias
data_lgs_quantized = np.clip(data_lgs_noisy_adu, 0, pco_edge_55.max_adu).astype(int)  # Cuantizar y limitar al máximo ADU   
safe_data_lgs_large  = np.maximum(data_lgs_large, 0)
data_lgs_large_shot  = np.random.normal(safe_data_lgs_large, np.sqrt(safe_data_lgs_large))
data_lgs_large_shot += np.random.poisson(pco_edge_55.dark_current * t_exp, size=data_lgs_large.shape).astype(float)
data_lgs_large_shot += np.random.normal(0, pco_edge_55.read_noise, size=data_lgs_large.shape)
data_lgs_large_noisy_adu  = data_lgs_large_shot / pco_edge_55.gain + bias
data_lgs_large_quantized  = np.clip(data_lgs_large_noisy_adu, 0, pco_edge_55.max_adu).astype(int)

# Visualizar la imagen del penacho LGS y penacho LGS grande
# fig, axes = plt.subplots(1, 2, figsize=(15, 5))
# # Imagen del penacho LGS
# noise_floor_adu = bias + 3 * np.sqrt(pco_edge_55.read_noise**2 + pco_edge_55.dark_current * t_exp) / pco_edge_55.gain
# axes[0].imshow(data_lgs_quantized, origin='lower', cmap='viridis', norm=LogNorm(vmin=noise_floor_adu, vmax=pco_edge_55.max_adu))
# axes[0].set_title('Simulated LGS Plume Image (ADU)')
# axes[0].set_xlabel('X (px)')   
# axes[0].set_ylabel('Y (px)')

# axes[1].imshow(data_lgs_large_quantized, origin='lower', cmap='viridis', norm=LogNorm(vmin=noise_floor_adu, vmax=pco_edge_55.max_adu))
# axes[1].set_title('Simulated Large LGS Plume Image (ADU)')
# axes[1].set_xlabel('X (px)')
# axes[1].set_ylabel('Y (px)')
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(6, 5))  
# plt.imshow(data_lgs, origin='lower', cmap='viridis')
# plt.title('Simulated LGS Plume Image (ADU)')
# plt.xlabel('X (px)')
# plt.ylabel('Y (px)')
# plt.colorbar(label='ADU')
# plt.tight_layout()
# plt.show()


