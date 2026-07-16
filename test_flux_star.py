import os
from objects import *
from astropy.table import Table
import matplotlib.pyplot as plt
import requests
from io import BytesIO



# ── Check SED of test star from VizieR Photometry Viewer ─────────────────────
star_test = star("Vega", mag=0.03, temp=9602)
B_mag = 0.03  # B-band magnitude of star_test


# Parameters to start the query
target = star_test.name
radius = 5   # arcsec

url = f"https://vizier.cds.unistra.fr/viz-bin/sed?-c={target}&-c.rs={radius}"
print(f"Querying: {url}")
response = requests.get(url, timeout=60)
response.raise_for_status()
sed = Table.read(BytesIO(response.content))

# Convert frequency (GHz) to wavelength (nm)
h = 6.62607015e-34    # Planck's constant (J*s)
c = 299792458 

freq_hz = sed["sed_freq"] * 1e9           # GHz → Hz
wl_m    = c / freq_hz         # m
wl_nm   = wl_m * 1e9                      # nm

# Flux: sed_flux in Jy → W/m²/nm
# F_λ [W/m²/nm] = F_ν [Jy] * 1e-26 * c / λ²[m] / 1e9
flux_si = sed["sed_flux"] * 1e-26 * c/ (wl_m**2) / 1e9

# ── Zero points (W/m²/nm) —
ZERO_POINTS = {
    'B':  {'wl_c':  440, 'F0': magnitude_2_flux(B_mag, F0=6.32e-11 )},
    'V':  {'wl_c':  550, 'F0': magnitude_2_flux(star_test.magnitude, F0=3.62286e-11 )}
}

# ── Planck model — in W/m²/nm ───────────────
wl_model  = np.arange(100, 30000, 10, dtype=float)   # nm
c_norm    = (np.trapezoid(planck_law(lambda_V*1e-9, star_test.T)*trans_V, lambda_V) /
             np.trapezoid(trans_V, lambda_V))
C         = star_test.flux / c_norm
flux_planck = planck_law(wl_model*1e-9, star_test.T) * C   # W/m²/nm

# ── Gráfico ───────────────────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18})

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left panel: frequency vs Jy (as shown in VizieR's photometry viewer)
ax1 = axes[0]
ax1.scatter(sed["sed_freq"], sed["sed_flux"], s=40, color='#F4A261',
            edgecolors='black', linewidths=0.5, zorder=3)
ax1.set_xlabel('Frequency (GHz)')
ax1.set_ylabel('Flux (Jy)')
ax1.set_title(f'SED — {target}  (VizieR photometry)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, linestyle='--', alpha=0.4)

# Panel derecho: longitud de onda vs W/m²/nm + modelo Planck
ax2 = axes[1]
wl_min, wl_max = 300, 1100   # nm
mask = (flux_si > 0) & np.isfinite(flux_si) & (wl_nm >= wl_min) & (wl_nm <= wl_max)
# Ordenar por longitud de onda para que el plot sea continuo
sort_idx = np.argsort(wl_nm[mask])
ax2.plot(wl_nm[mask][sort_idx], flux_si[mask][sort_idx], color='#F4A261',
         linewidth=1.5, zorder=3, label='VizieR SED')
model_mask = (wl_model >= wl_min) & (wl_model <= wl_max)
ax2.plot(wl_model[model_mask], flux_planck[model_mask], color='#2EC4B6', linewidth=2,
         label='Planck model (Teff=9602 K)')
# Zero points como referencia de calibración
zp_colors = {'B': '#4361EE', 'V': '#2DC653', 'J': '#E76F51', 'H': '#9D4EDD', 'Ks': '#F4A261'}
for band, zp in ZERO_POINTS.items():
    ax2.scatter(zp['wl_c'], zp['F0'], s=120, marker='*', zorder=5,
                color=zp_colors[band], edgecolors='black', linewidths=0.5,
                label=f'{band} ZP')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Flux density (W/m²/nm)')
ax2.set_title(f'SED vs Planck model — {target}')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()




def get_filter_svo(filter_id: str) -> tuple:
    """
Returns the transmission curve of a filter.
Looks first for a local VOTable file in filtro/<ID_with_dots>.xml;
if not found, downloads from the SVO Filter Profile Service.

filter_id examples:
    "GAIA/GAIA3.G"
    "GAIA/GAIA3.Gbp"
    "GAIA/GAIA3.Grp"
    "Generic/Johnson.V"

Returns: (lambda_m [m], transmission [0-1])
"""
    lam_list = []
    R_list   = []

    local_name = filter_id.replace("/", ".") + ".xml"
    local_path = os.path.join("filtro", local_name)

    if os.path.exists(local_path):
        root = ET.parse(local_path).getroot()
        ns   = root.tag[1:root.tag.index("}")] if "}" in root.tag else ""
        p    = f"{{{ns}}}" if ns else ""
        for tr in root.iter(f"{p}TR"):
            tds = list(tr)
            if len(tds) >= 2:
                lam_list.append(float(tds[0].text))
                R_list.append(float(tds[1].text))

    lam_m = np.array(lam_list) * 1e-10   # Angstrom → metres
    R     = np.array(R_list)
    return lam_m, R