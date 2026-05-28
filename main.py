import numpy as np
import astropy.units as u
import astropy.constants as const

# -----------------------------------------------
# Sistema AB: flujo de referencia en banda V
# F0 = 3631 Jy (1 Jy = 1e-26 W/m²/Hz)
# -----------------------------------------------

def magnitud_a_flujo(mag, F0=3631 * u.Jy):
    """
    Convierte magnitud aparente (sistema AB) a flujo por unidad de frecuencia en Jy.
    """
    flujo = F0 * 10**(-0.4 * mag)
    return flujo.to(u.W / u.m**2 / u.Hz)

# Ejemplo: estrella con magnitud V = 10
mag = 10.0
flujo_hz = magnitud_a_flujo(mag)
print(flujo_hz)

# Convertir a flujo por longitud de onda (W/m²/Å)
lam_central = 5500 * u.AA  # banda V
flujo_lam = (flujo_hz * const.c / lam_central**2).to(u.W / u.m**2 / u.AA)
print(f"Flujo (por Angstrom): {flujo_lam:.3e}")

# -----------------------------------------------
# Parámetros del sistema óptico
# -----------------------------------------------

diametro     = 0.3 * u.m       # diámetro del telescopio
oscurecimiento = 0            # fracción obstruida por araña/espejo secundario
eta          = 0.75             # eficiencia cuántica del sensor (75%)
# transmision  = 0.85             # transmisión óptica total (lentes + filtro)
ancho_banda  = 4000 * u.AA      # ancho de banda del filtro fotometrico V, sino se usa filtro se 
                                # entonces esto depende de la curva de eficiencia cuántica por ahora 
                                # se asume como todo el rango visible
t_exp        = 60 * u.s         # tiempo de exposición

# Área efectiva del telescopio
A = np.pi * (diametro / 2)**2 * (1 - oscurecimiento)
print(f"Área efectiva: {A.to(u.cm**2):.4f}")

# Energía por fotón en la longitud de onda central
E_foton = (const.h * const.c / lam_central).to(u.J)
print(f"Energía por fotón: {E_foton:.3e}")

# Flujo total integrado en la banda (W/m²)
flujo_total = (flujo_lam * ancho_banda).to(u.W / u.m**2)
print(f"Flujo integrado en banda V: {flujo_total:.3e}")

# Fotones detectados
N_fotones = (flujo_total * A * eta * t_exp / E_foton).decompose()
print(f"\n⭐ Fotones detectados: {N_fotones:.4e}")

# -----------------------------------------------
# Parámetros del detector CCD/CMOS
# -----------------------------------------------

ganancia     = 1.5              # e⁻/ADU (gain del sensor)
ruido_lectura = 10.0            # electrones RMS de ruido de lectura
dark_current = 0.01 * u.electron / u.s  # corriente oscura
pixeles_apertura = 25           # píxeles dentro de la apertura fotométrica

# Señal en electrones
signal_e = N_fotones.value  # ya está en electrones (η incluido)

# Ruido de Poisson de la señal
ruido_poisson = np.sqrt(signal_e)

# Ruido de corriente oscura
ruido_dark = np.sqrt((dark_current * t_exp).value * pixeles_apertura)

# Ruido de lectura total
ruido_read = ruido_lectura * np.sqrt(pixeles_apertura)

# Ruido total (en cuadratura)
ruido_total = np.sqrt(ruido_poisson**2 + ruido_dark**2 + ruido_read**2)

# Conversión a ADU
signal_adu = signal_e    / ganancia
ruido_adu  = ruido_total / ganancia

# Relación señal/ruido
snr = signal_e / ruido_total

print(f"\n📊 Resultados del sensor:")
print(f"   Señal:              {signal_e:.2e} e⁻")
print(f"   Señal en ADU:       {signal_adu:.2e} ADU")
print(f"   Ruido Poisson:      {ruido_poisson:.2f} e⁻")
print(f"   Ruido dark:         {ruido_dark:.2f} e⁻")
print(f"   Ruido lectura:      {ruido_read:.2f} e⁻")
print(f"   Ruido total:        {ruido_total:.2f} e⁻")
print(f"   SNR:                {snr:.1f}")

import matplotlib.pyplot as plt

def calcular_snr(mag, diametro=0.3, t_exp=60, eta=0.75):
    F0 = 3631 * u.Jy
    lam = 5500 * u.AA
    bw  = 890 * u.AA
    
    flujo  = (F0 * 10**(-0.4 * mag) * const.c / lam**2).to(u.W / u.m**2 / u.AA)
    A      = np.pi * (diametro * u.m / 2)**2 * 0.9
    E_fot  = (const.h * const.c / lam).to(u.J)
    N      = (flujo * bw * A * 0.85 * eta * t_exp * u.s / E_fot).decompose().value
    
    ruido  = np.sqrt(N + 0.01 * t_exp * 25 + (10 * np.sqrt(25))**2)
    return N / ruido

magnitudes = np.linspace(5, 18, 100)
snr_vals   = [calcular_snr(m) for m in magnitudes]

plt.figure(figsize=(9, 5))
plt.semilogy(magnitudes, snr_vals, color='steelblue', lw=2)
plt.axhline(y=5,  color='red',    linestyle='--', label='SNR = 5 (límite detección)')
plt.axhline(y=100, color='green', linestyle='--', label='SNR = 100 (fotometría precisa)')
plt.xlabel('Magnitud aparente (V)', fontsize=12)
plt.ylabel('SNR', fontsize=12)
plt.title('Relación Señal/Ruido vs Magnitud (D=30cm, t=60s)', fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()  # magnitudes crecen hacia la derecha = más débil
plt.tight_layout()
plt.show()