import os
import h5py
import re
import numpy as np
import matplotlib
# Configuración para ambiente sin interfaz gráfica
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage import gaussian_filter1d

# Constante física para la corrección de Energía Magnética
FOUR_PI = 4.0 * np.pi

# ==============================
# FUNCIONES AUXILIARES
# ==============================

def explorar_archivo(file_path):
    """Explora y lista los datasets de un archivo HDF5."""
    with h5py.File(file_path, "r") as f:
        print("\n=== Explorando archivo ===")
        def recorrer(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Grupo: {name}")
        f.visititems(recorrer)
        print("==========================\n")


def time_from_snapshot(file_path):
    """Extrae el tiempo de simulación del archivo HDF5 de forma robusta."""
    with h5py.File(file_path, "r") as f:
        # 1. Buscar en 'real scalars'
        if "real scalars" in f:
            group = f["real scalars"]
            if isinstance(group, h5py.Dataset) and "time" in group.dtype.names:
                return group["time"][()][0] # Lee el primer elemento del dataset
        
        # 2. Buscar en 'simulation parameters'
        if "simulation parameters" in f and "time" in f["simulation parameters"].attrs:
            return f["simulation parameters"].attrs["time"]

        # 3. Buscar en atributos del archivo raíz
        if "time" in f.attrs:
            return f.attrs["time"]
    return None


def read_beta_from_par(par_file):
    """Lee el parámetro BETA del flash.par."""
    beta_value = None
    if not os.path.exists(par_file):
        return None
    with open(par_file, "r") as f:
        for line in f:
            if "BETA" in line.upper():
                try:
                    # Intenta convertir a float, reemplazando 'd' por 'e' (notación de Fortran)
                    beta_value = float(line.split("=")[1].strip().replace("d", "e"))
                except:
                    # Si falla, se queda como string sin intentar la conversión
                    beta_value = line.split("=")[1].strip()
                break
    return beta_value


def load_snapshot_density(file_path, var="dens"):
    """Carga los datos de densidad (o cualquier otra var) y la caja delimitadora."""
    with h5py.File(file_path, "r") as f:
        data = f[var][:]
        bbox = f["bounding box"][:]
    blocks = []
    for i in range(data.shape[0]):
        x0, x1 = bbox[i, 0]
        y0, y1 = bbox[i, 1]
        nx, ny = data.shape[2], data.shape[3]
        x = np.linspace(x0, x1, nx)
        y = np.linspace(y0, y1, ny)
        vals = data[i, 0, :, :] 
        blocks.append((x, y, vals))
    return blocks


def compute_disk_height(blocks, rmin=3.0, rmax=4.7):
    """Calcula la altura H del disco (desviación estándar ponderada por densidad)."""
    rs, zs, rho = [], [], []
    for (x, y, vals) in blocks:
        X, Y = np.meshgrid(x, y, indexing="ij")
        R = X
        mask = (R >= rmin) & (R <= rmax)
        rs.extend(R[mask].flatten())
        zs.extend(Y[mask].flatten())
        rho.extend(vals[mask].flatten())
    rs, zs, rho = np.array(rs), np.array(zs), np.array(rho)
    if rho.sum() == 0:
        return 0.0
    z_mean = np.average(zs, weights=rho)
    z2_mean = np.average(zs**2, weights=rho)
    # H = sqrt(<z^2> - <z>^2)
    return np.sqrt(z2_mean - z_mean**2)


def compute_density_profile(blocks, rmin=3.0, rmax=4.7, nbins=200):
    """Calcula el perfil de probabilidad de densidad P(z)."""
    rs, zs, rho = [], [], []
    for (x, y, vals) in blocks:
        X, Y = np.meshgrid(x, y, indexing="ij")
        R = X
        mask = (R >= rmin) & (R <= rmax)
        rs.extend(R[mask].flatten())
        zs.extend(Y[mask].flatten())
        rho.extend(vals[mask].flatten())
    zs, rho = np.array(zs), np.array(rho)
    if rho.sum() == 0:
        return None, None
    hist, edges = np.histogram(zs, bins=nbins, weights=rho, density=True)
    z_centers = 0.5 * (edges[1:] + edges[:-1])
    hist = gaussian_filter1d(hist, sigma=1.5)
    return z_centers, hist


def compute_H_and_profiles(folder, rmin=3.0, rmax=4.7, var="dens"):
    """Calcula la altura H y la Kurtosis para todos los checkpoints."""
    archivos = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.startswith("torus_mhd_2d_hdf5_chk")
    ])
    tiempos, alturas, kurtosis_vals, perfiles = [], [], [], []

    for archivo in archivos:
        try:
            print(f"Procesando H y Kurtosis de {os.path.basename(archivo)}...")
            blocks = load_snapshot_density(archivo, var=var)
            H = compute_disk_height(blocks, rmin=rmin, rmax=rmax)
            zc, hist = compute_density_profile(blocks, rmin=rmin, rmax=rmax)
            if zc is None or hist is None:
                continue

            # Momentos estadísticos usando np.trapezoid
            mean_z = np.trapezoid(zc * hist, zc)
            var_z = np.trapezoid((zc - mean_z)**2 * hist, zc)
            kurt_z = np.trapezoid((zc - mean_z)**4 * hist, zc) / (var_z**2)

            t = time_from_snapshot(archivo)
            # Asegurar que 't' es numérico (float o int)
            if t is None:
                t = len(tiempos) 

            tiempos.append(t)
            alturas.append(H)
            kurtosis_vals.append(kurt_z)
            perfiles.append((zc, hist))

        except Exception as e:
            print(f"❌ Error con {os.path.basename(archivo)}: {e}")

    return np.array(tiempos), np.array(alturas), np.array(kurtosis_vals), perfiles


# --- Funciones para el CÁLCULO DE ENERGÍAS ---

def leer_parametros_flashpar(ruta_par):
    """Lee parámetros clave del flash.par."""
    params = {}
    if not os.path.exists(ruta_par):
        # Valores por defecto para la simulación de toro
        return {"gamma": 5.0 / 3.0, "gravity_constant": 1.0, "ptmass": 1.0, "R_Sphere": 1.5}
    with open(ruta_par, "r") as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                k, v = [x.strip() for x in line.split("=", 1)]
                v = v.split()[0]
                try:
                    params[k] = float(v.replace("d", "e"))
                except ValueError:
                    params[k] = v
    return params

def phi_paczynski_wiita(R, Z, G, M, R_S):
    """Calcula el potencial gravitatorio de Paczynski-Wiita."""
    r = np.sqrt(R**2 + Z**2)
    r = np.maximum(r, R_S + 1e-6) # Evita la singularidad
    return -G * M / (r - R_S)

def leer_variable_flash(f, nombre):
    """Busca una variable dentro del archivo FLASH, asegurando la forma (nblocks, 1, nx, ny)."""
    if nombre in f:
        return f[nombre][:]
    if "unknown names" in f and "data" in f:
        names = [n.decode("utf-8").strip() for n in f["unknown names"][:]]
        if nombre in names:
            idx = names.index(nombre)
            data_packed = f["data"][:, idx, :, :] # (nblocks, nx, ny)
            # Retorna con la dimensión extra, compatible con la forma de un _chk file
            return data_packed[:, np.newaxis, :, :] 
    raise KeyError(f"Variable {nombre} no encontrada en {f.filename}")


def compute_energies(folder, gamma, G, M, R_S):
    """Calcula energías total, cinética, magnética, térmica y gravitatoria
       en coordenadas cilíndricas (R,Z) correctamente."""
    
    archivos = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.startswith("torus_mhd_2d_hdf5_chk")
    ])

    tiempos, E_kin, E_mag, E_th, E_grav = [], [], [], [], []

    for archivo in archivos:
        print(f"Calculando energías de {os.path.basename(archivo)}...")

        Ekin = Emag = Eth = Egrav = 0.0

        try:
            with h5py.File(archivo, "r") as f:

                rho  = leer_variable_flash(f, "dens")
                pres = leer_variable_flash(f, "pres")

                vx = leer_variable_flash(f, "velx")
                vy = leer_variable_flash(f, "vely")
                vz = leer_variable_flash(f, "velz") if "velz" in f else np.zeros_like(vx)

                Bx = leer_variable_flash(f, "magx")
                By = leer_variable_flash(f, "magy")
                Bz = leer_variable_flash(f, "magz") if "magz" in f else np.zeros_like(Bx)

                bbox = f["bounding box"][:]  # (nblocks, 2, 2)

                for i in range(rho.shape[0]):

                    # --- límites del bloque ---
                    R_min, R_max = bbox[i, 0, 0], bbox[i, 0, 1]
                    Z_min, Z_max = bbox[i, 1, 0], bbox[i, 1, 1]

                    # --- celdas ---
                    nx, ny = rho.shape[2], rho.shape[3]

                    # Diferenciales correctos del grid de FLASH (tamaño de celda)
                    dR = (R_max - R_min) / (nx - 1)
                    dZ = (Z_max - Z_min) / (ny - 1)

                    # Centros de celda (geometría FLASH)
                    R = R_min + (np.arange(nx) + 0.5) * dR
                    Z = Z_min + (np.arange(ny) + 0.5) * dZ

                    RR, ZZ = np.meshgrid(R, Z, indexing="ij")

                    # --- variables del bloque ---
                    rho_b  = rho[i, 0]
                    pres_b = pres[i, 0]
                    vx_b, vy_b, vz_b = vx[i, 0], vy[i, 0], vz[i, 0]
                    Bx_b, By_b, Bz_b = Bx[i, 0], By[i, 0], Bz[i, 0]

                    # --- energías ---
                    v2   = vx_b**2 + vy_b**2 + vz_b**2
                    e_kin = 0.5 * rho_b * v2
                    e_mag = (Bx_b**2 + By_b**2 + Bz_b**2) / (8*np.pi)
                    e_th  = pres_b / (gamma - 1.0)

                    phi = phi_paczynski_wiita(RR, ZZ, G, M, R_S)
                    e_grav = rho_b * phi

                    # --- INTEGRACIÓN CORRECTA ---
                    peso = 2 * np.pi * RR * dR * dZ

                    Ekin  += np.sum(e_kin * peso)
                    Emag  += np.sum(e_mag * peso)
                    Eth   += np.sum(e_th  * peso)
                    Egrav += np.sum(e_grav * peso)

                # tiempo del snapshot
                t = time_from_snapshot(archivo)
                if t is None: t = len(tiempos)

                tiempos.append(t)
                E_kin.append(Ekin)
                E_mag.append(Emag)
                E_th.append(Eth)
                E_grav.append(Egrav)

        except Exception as e:
            print(f"❌ Error al calcular energías en {os.path.basename(archivo)}: {e}")

    tiempos = np.array(tiempos)
    order = np.argsort(tiempos)

    return (
        tiempos[order],
        np.array(E_kin)[order],
        np.array(E_mag)[order],
        np.array(E_th)[order],
        np.array(E_grav)[order],
    )



# ==============================
# MAIN (Con la modificación de la cuadrícula de energías)
# ==============================

if __name__ == "__main__":
    # --- DIRECTORIOS Y ARCHIVOS ---
    # NOTA: Cambia esta ruta si es necesario para tu estructura de archivos
    carpeta = "/mnt/data/FLASH4.8/TorusTest2/data"
    par_file = os.path.join(carpeta, "flash.par")

    ejemplo = os.path.join(carpeta, "torus_mhd_2d_hdf5_chk_0050")
    try:
        explorar_archivo(ejemplo)
    except Exception as e:
        print(f"⚠️ No se pudo explorar el archivo de ejemplo: {e}")

    # Cargar parámetros globales
    params = leer_parametros_flashpar(par_file)
    gamma = params.get("gamma", 5.0 / 3.0)
    G = params.get("gravity_constant", 1.0)
    M = params.get("ptmass", 1.0)
    R_S = params.get("R_Sphere", 1.5)
    
    beta_value = read_beta_from_par(par_file)

    # --- 1. Cálculo de H y Kurtosis ---
    tiempos, alturas, kurtosis_vals, perfiles = compute_H_and_profiles(carpeta)

    # ---- Gráfico H(t) y Kurtosis(t) en paneles separados ----
    if len(tiempos) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

        color1, color2 = "royalblue", "darkorange"

        # --- Panel izquierdo: H vs tiempo ---
        ax1.plot(tiempos, alturas, "-o", lw=1.2, ms=3, color=color1, alpha=0.9)
        ax1.set_xlabel("Tiempo")
        ax1.set_ylabel("H (grosor del disco)", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_title("Evolución de H(t)")
        ax1.grid(True, ls="--", alpha=0.6)

        # --- Panel derecho: Kurtosis vs tiempo ---
        ax2.plot(tiempos, kurtosis_vals, "-s", lw=1.2, ms=3, color=color2, alpha=0.9)
        ax2.axhline(3, color="gray", lw=0.8, ls="--", label="Gaussiana pura (k=3)")
        ax2.set_xlabel("Tiempo")
        ax2.set_ylabel("Kurtosis", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_title("Evolución de la Kurtosis(t)")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, ls="--", alpha=0.6)

        if beta_value is not None:
            fig.text(
                0.05, 0.95, f"BETA = {beta_value}",
                transform=fig.transFigure,
                fontsize=10, color="darkred",
                ha="left", va="top",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
            )

        plt.suptitle("Evolución del grosor y la curtosis del disco (FLASH MHD)", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_file = os.path.join(carpeta, f"H_y_kurtosis_paneles_beta{beta_value}.png")
        plt.savefig(output_file, dpi=250)
        print(f"✅ Gráfico guardado en: {output_file}")
    else:
        print("⚠️ No hay suficientes checkpoints para calcular H y Kurtosis.")


    # ---- 2. GIF de evolución P(z) ----
    frames = []
    gif_path = os.path.join(carpeta, f"evolucion_probabilidad_beta{beta_value}.gif")
    if perfiles:
        max_hist = max(np.max(hist) for _, hist in perfiles if hist is not None)
        z_global = np.concatenate([zc for zc, _ in perfiles if zc is not None])
        zmin, zmax = np.min(z_global), np.max(z_global)

        fig, ax = plt.subplots(figsize=(7, 5))

        for (zc, hist), t, H, k in zip(perfiles, tiempos, alturas, kurtosis_vals):
            if zc is None or hist is None:
                continue
            ax.clear()

            # --- Puntos + curva simulación ---
            ax.plot(zc, hist, "-", color="navy", lw=1.0, alpha=0.8, label="P(z) simulación")
            ax.scatter(zc, hist, s=14, color="royalblue", alpha=0.8, zorder=3)

            # --- Curva gaussiana teórica ---
            z_fit = np.linspace(zc.min(), zc.max(), 400)
            gauss = np.exp(-z_fit**2 / (2 * H**2))
            gauss /= np.trapezoid(gauss, z_fit)
            gauss *= np.trapezoid(hist, zc)
            ax.plot(z_fit, gauss, "--", color="darkred", lw=1.2, label="Gaussiana teórica")

            # --- Texto con ecuación y parámetros ---
            ax.text(
                0.05, 0.78,
                f"$P(z) \\propto e^{{-z^2 / (2H^2)}}$\n"
                f"$t = {t:.2f}$\n"
                f"$H = {H:.3e}$\n"
                f"$K = {k:.2f}$\n"
                f"$\\beta = {beta_value}$",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )

            ax.set_xlabel("z")
            ax.set_ylabel("P(z)")
            ax.set_ylim(0, max_hist * 1.2)
            ax.set_xlim(zmin, zmax)
            ax.grid(True, ls=":", alpha=0.7)
            ax.legend(loc="upper right", fontsize=9)
            ax.set_title("Evolución temporal de P(z) (FLASH MHD)")

            # --- Convertir frame a imagen y agregar al GIF ---
            fig.canvas.draw()
            image = np.array(fig.canvas.buffer_rgba())
            frames.append(image.copy())

        imageio.mimsave(gif_path, frames, fps=6)
        print(f"🎞 GIF guardado en: {gif_path}")
    else:
        print("⚠️ No se encontraron archivos de checkpoint válidos para generar el GIF P(z).")


    # --- 3. Cálculo y Gráfico de Energías ---
    print("\n=== Calculando energías del sistema (cinética, magnética, térmica, gravitatoria) ===")
    tiempos_E, E_kin, E_mag, E_th, E_grav = compute_energies(carpeta, gamma, G, M, R_S)
    
    E_total = E_kin + E_mag + E_th + E_grav

    if len(tiempos_E) > 0:
        plt.style.use("seaborn-v0_8-darkgrid")
        
        # Crear una figura con 3 filas y 2 columnas para organizar los 5 gráficos
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        # Usamos GridSpec para mejor control de la disposición: 3 filas, 2 columnas
        gs = fig.add_gridspec(3, 2)
        
        # Subgráficos 2x2 para las 4 energías principales
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        # Subgráfico 1x1 para la energía total (ocupa las dos columnas de la última fila)
        ax5 = fig.add_subplot(gs[2, :]) 
        
        axs = [ax1, ax2, ax3, ax4, ax5]
        
        # MODIFICACIÓN: Usar Joules [J] pero con la aclaración de que el valor
        # numérico es la magnitud en unidades de código (escala E+45)
        UNITS_LABEL = r" [J] ($10^{45}$ )"
        
        labels = [
            r"$E_{\rm kin}$" + UNITS_LABEL, 
            r"$E_{\rm mag}$" + UNITS_LABEL, 
            r"$E_{\rm th}$" + UNITS_LABEL, 
            r"$E_{\rm grav}$" + UNITS_LABEL, 
            r"$E_{\rm tot}$" + UNITS_LABEL
        ]
        
        datas = [E_kin, E_mag, E_th, E_grav, E_total]
        titles = ["CINÉTICA", "MAGNÉTICA", "TÉRMICA", "GRAVITATORIA", "TOTAL"]
        colors = ["royalblue", "darkred", "orange", "purple", "black"]
        
        # Obtener los límites del eje x (tiempo) para todos
        x_lim = (tiempos_E.min(), tiempos_E.max())
        
        # Se itera sobre los 5 ejes, pero solo se pone la etiqueta X en la última fila
        for i, (ax, data, label, title, color) in enumerate(zip(axs, datas, labels, titles, colors)):
            ax.plot(tiempos_E, data, lw=1.5, color=color)
            # Formato científico para el eje Y, para mejor lectura de magnitudes
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) 
            ax.set_ylabel(label)
            ax.set_title(f"Energía {title}", fontsize=10)
            ax.set_xlim(x_lim)
            ax.grid(True, ls="--", alpha=0.6)
            
            # Solo la última fila (índice 4) debe tener la etiqueta del eje X
            if i >= 4:
                ax.set_xlabel("Tiempo")

        plt.suptitle("Evolución de las Energías del Sistema (FLASH MHD)", fontsize=14, y=1.02)
        
        # Se cambia el nombre del archivo de salida para reflejar la nueva etiqueta
        out_path = os.path.join(carpeta, "energias_cuadricula_J.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"✅ Gráfico de energías guardado en: {out_path}")
    else:
        print("⚠️ No hay suficientes datos para generar el gráfico de energías.")