"""
Code zur Bachelorarbeit:Entwicklung und Anwendung numerischer Verfahren zur Simulation und Analyse
der Druckverteilung an umstroemten Koerpern
2D-SIMPLE-Solver (stationoer, laminar, inkompressibel)

Daniela Nguimkeng
952119

"""

import numpy as np
import matplotlib.pyplot as plt
from math import isfinite

# =========================
# 1) Pre-processing: Eingaben/ Gitter / Felder / Maske / Randbedingungen / Helfer
# =========================

def frage(prompt, default, cast=float):
    try:
        s = input(f"{prompt} [Default: {default}]: ").strip()
        return default if s == "" else cast(s)
    except Exception:
        print("Ungueltige Eingabe Default wird verwendet.")
        return default

def frage_wahl(prompt, choices, default):
    s = input(f"{prompt} {choices} [Default: {default}]: ").strip().lower()
    return s if s in choices else default

def eingaben():
    print("=== Kanalstroemung mit Hindernis Eingaben (Enter = Default) ===")
    geom = frage_wahl("Hindernis (kreis/rechteck)", ["kreis", "rechteck"], "rechteck")
    Lx = frage("Kanal-Laenge Lx [m]", 0.8, float)
    Ly = frage("Kanal-Hoehe Ly [m]", 0.2, float)
    Nx = frage("Zellen in x (Nx)", 320, int)
    Ny = frage("Zellen in y (Ny)", 160, int)
    U_in = frage("Einlassgeschwindigkeit U_in [m/s]", 1.0, float)
    rho = frage("Dichte rho [kg/m^3]", 1.0, float)

    if geom == "kreis":
        cx = frage("Zentrum x_c [m]", 0.1125, float)
        cy = frage("Zentrum y_c [m]", 0.1, float)
        R  = frage("Radius R [m]",    0.0125, float)
        obst = ("circle", (cx, cy, R))
        Lref = 2.0 * R
    else:
        x0 = frage("Block x0 [m]", 0.1, float)
        y0 = frage("Block y0 [m]", 0.075, float)
        w  = frage("Block-Breite w [m]", 0.025, float)
        h  = frage("Block-Hoehe h [m]",   0.025, float)
        obst = ("rect", (x0, y0, w, h))
        Lref = w

    Re = frage("Reynolds-Zahl Re (laminar waehlen)", 40.0, float)
    Re = max(Re, 1e-12)
    mu = rho * U_in * Lref / Re

    scheme   = "upwind"
    n_outer  = frage("Max. Outer-Iterationen", 750, int)
    tol      = frage("Toleranz (Gleichungsresiduen-Abbruch)", 1e-6, float)
    alpha_u  = frage("Unterrelaxation alpha_u (0-1)", 0.3, float)
    alpha_p  = frage("Unterrelaxation alpha_p (0-1)", 0.3, float)
    omega    = 1.0

    cfg = dict(
        geom=geom, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, U_in=U_in, rho=rho,
        mu=mu, Re=Re, Lref=Lref, obst=obst, scheme=scheme,
        n_outer=n_outer, tol=tol, alpha_u=alpha_u, alpha_p=alpha_p, omega=omega,
        M_in=rho*U_in*Ly  # Einlass-Massenstrom (kg/s pro Einheit Tiefe)
    )
    print("Eingaben uebernommen:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    return cfg

# Gitter / Felder / Maske

def gitter(Lx,Ly,Nx,Ny):
    dx = Lx / Nx
    dy = Ly / Ny
    x_w = np.linspace(0.0, Lx, Nx+1)
    y_w = np.linspace(0.0, Ly, Ny+1)
    x_c = 0.5*(x_w[:-1] + x_w[1:])
    y_c = 0.5*(y_w[:-1] + y_w[1:])
    Xc, Yc = np.meshgrid(x_c, y_c)
    return x_c, y_c, Xc, Yc, x_w, y_w, dx, dy

def plot_mesh(x_w, y_w, obst=None):
    """Netzlinien ueber das gesamte Gebiet."""
    plt.figure(figsize=(7, 2.2))
    for xv in x_w:
        plt.plot([xv, xv], [y_w[0], y_w[-1]], color='0.85', linewidth=0.5)
    for yv in y_w:
        plt.plot([x_w[0], x_w[-1]], [yv, yv], color='0.85', linewidth=0.5)
    # Hindernis umranden 
    if obst and obst[0] == "circle":
        cx, cy, R = obst[1]
        th = np.linspace(0, 2*np.pi, 361)
        plt.plot(cx + R*np.cos(th), cy + R*np.sin(th), 'k', linewidth=1.5)
    elif obst and obst[0] == "rect":
        x0, y0, w, h = obst[1]
        xs = [x0, x0+w, x0+w, x0, x0]; ys = [y0, y0, y0+h, y0+h, y0]
        plt.plot(xs, ys, 'k', linewidth=1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(x_w[0], x_w[-1]); plt.ylim(y_w[0], y_w[-1])
    plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.title('Gitter (Netzlinien)')
    plt.tight_layout()
    plt.show()

def zeichne_hindernis(ax, obst):
    if obst and obst[0] == "circle":
        cx, cy, R = obst[1]
        th = np.linspace(0, 2*np.pi, 361)
        ax.plot(cx + R*np.cos(th), cy + R*np.sin(th), 'k', linewidth=1.8)
    elif obst and obst[0] == "rect":
        x0, y0, w, h = obst[1]
        xs = [x0, x0+w, x0+w, x0, x0]
        ys = [y0, y0, y0+h, y0+h, y0]
        ax.plot(xs, ys, 'k', linewidth=1.8)
        
def felder(Nx,Ny):
    u = np.zeros((Ny, Nx))
    v = np.zeros((Ny, Nx))
    p = np.zeros((Ny, Nx))
    return u, v, p

def fluid_maske(Xc, Yc, obst):
    """True = Fluid, False = Festkoerper."""
    mask = np.ones_like(Xc, dtype=bool)
    if obst[0] == "circle":
        cx, cy, R = obst[1]
        mask &= ((Xc - cx)**2 + (Yc - cy)**2) >= R**2
    else:
        x0, y0, w, h = obst[1]
        mask &= ~((Xc >= x0) & (Xc <= x0 + w) & (Yc >= y0) & (Yc <= y0 + h))
    return mask

# Randbedingungen
def randbedingung_uv(u, v, cfg, fluid):
    # Inlet
    u[:, 0] = cfg["U_in"]; v[:, 0] = 0.0
    # Outlet (Neumann ~ Kopie)
    u[:, -1] = u[:, -2]; v[:, -1] = v[:, -2]
    # Waende
    u[0, :], u[-1, :] = 0.0, 0.0
    v[0, :], v[-1, :] = 0.0, 0.0
    # Hindernis
    u[~fluid] = 0.0; v[~fluid] = 0.0
    return u, v

def randbedingung_p(p):
    p[0, 0] = 0.0
    p[:, -1] = p[:, -2]
    
    return p

# =========================
# 2) Solver: Gauss-Seidel/ Diskretisierung Impuls/ KrŠfte, c_D und c_P/SIMPLE
# =========================

def gauss_seidel(phi, aE, aW, aN, aS, aP, b, fluid, itmax=200, tol=1e-8, omega=1.0):
    """
    Einfache GS-Iteration mit Sicherheitspruefungen.
    omega = 1.0 -> normal; <1.0 -> zusaetzliche Daempfung
    """
    Ny, Nx = phi.shape
    it = 0
    max_res = 1e12
    while it < itmax and max_res > tol:
       max_res = 0.0
       for j in range(1, Ny-1):
           for i in range(1, Nx-1):
               if not fluid[j, i]:
                   phi[j, i] = 0.0
                   continue
               rhs = (aE[j, i]*phi[j, i+1] + aW[j, i]*phi[j, i-1]  #vgl Gleichung 3.5.1
                      + aN[j, i]*phi[j-1, i] + aS[j, i]*phi[j+1, i] + b[j, i])
               denom = aP[j, i] + 1e-30
               phi_new = rhs / denom
               # Update
               phi[j, i] = phi_new if np.isfinite(phi_new) else 0.0
               # Gleichungsrest (lokal)
               res = abs(rhs - aP[j, i]*phi[j, i])
               if not isfinite(res): res = 1e12
               if res > max_res: max_res = res
       it += 1
    return max_res, it

# Diskretisierung Impuls (Konvektion + Diffusion)
def momentum(u, v, p, rho, mu, dx, dy, fluid):
    """
    Liefert Koeffizienten und RHS fuer u- und v-Gleichung:
    aE,aW,aN,aS,aP,b  (jeweils einmal fuer u und v)
    """
    Ny, Nx = u.shape
    Ae = Aw = dy
    An = As = dx
    vol = dx * dy 

    def zeros_like_u():
        return (np.zeros_like(u), np.zeros_like(u), np.zeros_like(u),
                np.zeros_like(u), np.zeros_like(u), np.zeros_like(u)) 

    aEu, aWu, aNu, aSu, aPu, bu = zeros_like_u()
    aEv, aWv, aNv, aSv, aPv, bv = zeros_like_u()

    for j in range(Ny):
        for i in range(Nx):
            if not fluid[j, i]:
                aPu[j, i] = 1.0; aPv[j, i] = 1.0
                continue

            # Diffusion
            De = mu * Ae / dx; Dw = mu * Aw / dx  
            Dn = mu * An / dy; Ds = mu * As / dy

            # Flaechenwerte (arithm. Mittel)
            ue = 0.5 * (u[j, i] + (u[j, i + 1] if i + 1 < Nx else u[j, i]))
            uw = 0.5 * (u[j, i] + (u[j, i - 1] if i - 1 >= 0 else u[j, i]))
            vn = 0.5 * (v[j, i] + (v[j - 1, i] if j - 1 >= 0 else v[j, i]))
            vs = 0.5 * (v[j, i] + (v[j + 1, i] if j + 1 < Ny else v[j, i]))

            # Konvektive Massenfluesse
            Fe = rho * ue * Ae; Fw = rho * uw * Aw
            Fn = rho * vn * An; Fs = rho * vs * As

            # Druckgradient (zentral-differenziert)
            dpdx = (p[j, i + 1] - p[j, i - 1]) / (2 * dx) if 0 < i < Nx - 1 else 0.0
            dpdy = (p[j + 1, i] - p[j - 1, i]) / (2 * dy) if 0 < j < Ny - 1 else 0.0
            bu[j, i] = -dpdx * vol
            bv[j, i] = -dpdy * vol

            # Upwind-Koeffizienten (bounded)
            aEu[j, i] = De + max(-Fe, 0.0)
            aWu[j, i] = Dw + max( Fw, 0.0)
            aNu[j, i] = Dn + max(-Fn, 0.0)
            aSu[j, i] = Ds + max( Fs, 0.0)
            aPu[j, i] = aEu[j, i] + aWu[j, i] + aNu[j, i] + aSu[j, i] + (Fe - Fw + Fn - Fs) + 1e-30 # vgl. Gleichung 3.1.3 (analog fŸr v)

            aEv[j, i] = De + max(-Fe, 0.0)
            aWv[j, i] = Dw + max( Fw, 0.0)
            aNv[j, i] = Dn + max(-Fn, 0.0)
            aSv[j, i] = Ds + max( Fs, 0.0)
            aPv[j, i] = aEv[j, i] + aWv[j, i] + aNv[j, i] + aSv[j, i] + (Fe - Fw + Fn - Fs) + 1e-30

            # Festkoerper-Nachbarn -> Kopplung 0 
            if i + 1 < Nx and not fluid[j, i + 1]: aEu[j, i] = 0.0; aEv[j, i] = 0.0
            if i - 1 >= 0 and not fluid[j, i - 1]: aWu[j, i] = 0.0; aWv[j, i] = 0.0
            if j - 1 >= 0 and not fluid[j - 1, i]: aNu[j, i] = 0.0; aNv[j, i] = 0.0
            if j + 1 < Ny and not fluid[j + 1, i]: aSu[j, i] = 0.0; aSv[j, i] = 0.0

    return (aEu, aWu, aNu, aSu, aPu, bu), (aEv, aWv, aNv, aSv, aPv, bv)

# FlŸsse / Kontinuitaet
def fluesse(u, v, cfg, dx, dy):
    """
    Konservative Fluesse aus arithm. Flaechengeschwindigkeiten.
    diskrete Kontinuitaet (vgl. Gleichung 3.1.2)
    
    """
    rho = cfg["rho"]; Ny, Nx = u.shape
    ue = 0.5 * (u[:, :-1] + u[:, 1:])   # Ny x (Nx-1)
    vn = 0.5 * (v[:-1, :] + v[1:, :])   # (Ny-1) x Nx

    Fe = np.zeros((Ny, Nx)); Fw = np.zeros((Ny, Nx))
    Fn = np.zeros((Ny, Nx)); Fs = np.zeros((Ny, Nx))

    Fe[:, :-1] = rho * ue * dy
    Fw[:,  1:] = rho * ue * dy
    Fn[:-1, :] = rho * vn * dx
    Fs[ 1:, :] = rho * vn * dx

    return Fe, Fw, Fn, Fs

# Kraefte, c_D und c_P
def kraefte_und_cd(u, v, p, cfg, grid, fluid, n_theta=720, mit_scherung=True):
    xc, yc, Xc, Yc, dx, dy = grid
    Ny, Nx = u.shape
    rho, U, mu = cfg["rho"], cfg["U_in"], cfg["mu"]

    p_inf = float(np.nanmean(p[:, -1]))  # Referenzdruck am Outlet
    Fx_p, Fx_tau = 0.0, 0.0
    Cp_punkte = []

    if cfg["obst"][0] == "circle":
        cx, cy, R = cfg["obst"][1]
        th = np.linspace(0.0, 2*np.pi, n_theta, endpoint=False)
        ct, st = np.cos(th), np.sin(th)
        xb, yb = cx + R*ct, cy + R*st
        # naechstliegende Zellen
        ii = np.clip(np.searchsorted(xc, xb) - 1, 0, Nx - 1)
        jj = np.clip(np.searchsorted(yc, yb) - 1, 0, Ny - 1)
        p_b = p[jj, ii]
        ds = R * (2*np.pi / n_theta)

        Fx_p -= np.sum((p_b - p_inf) * ct * ds)
        Cp = (p_b - p_inf) / (0.5 * rho * U**2)
        Cp_punkte = list(zip(th, Cp))

        if mit_scherung:
            delta = max(min(dx, dy), 1e-12)
            u_t = u[jj, ii]*(-st) + v[jj, ii]*(ct)  # tangential
            tau_t = mu * (u_t / delta)
            Fx_tau -= np.sum(tau_t * (-st) * ds)

        Aref = 2.0 * R

    else:
        x0, y0, w, h = cfg["obst"][1]
        eps = 1e-12
        left   = (np.abs(Xc - x0)     <= 0.5*dx + eps) & (Yc >= y0 - 0.5*dy - eps) & (Yc <= y0 + h + 0.5*dy + eps) & fluid
        right  = (np.abs(Xc - (x0+w)) <= 0.5*dx + eps) & (Yc >= y0 - 0.5*dy - eps) & (Yc <= y0 + h + 0.5*dy + eps) & fluid
        bottom = (np.abs(Yc - y0)     <= 0.5*dy + eps) & (Xc >= x0 - 0.5*dx - eps) & (Xc <= x0 + w + 0.5*dx + eps) & fluid
        top    = (np.abs(Yc - (y0+h)) <= 0.5*dy + eps) & (Xc >= x0 - 0.5*dx - eps) & (Xc <= x0 + w + 0.5*dx + eps) & fluid

        js, is_ = np.where(left | right | bottom | top)
        for j, i in zip(js, is_):
            if left[j, i]:   nx, ny, ds = -1.0, 0.0, dy
            elif right[j, i]:nx, ny, ds =  1.0, 0.0, dy
            elif bottom[j,i]:nx, ny, ds =  0.0,-1.0, dx
            else:            nx, ny, ds =  0.0, 1.0, dx

            Fx_p -= (p[j, i] - p_inf) * nx * ds
            # Umfangskoordinate s entlang der Rechteckkontur (Start: unten links)
            x = Xc[j, i]
            y = Yc[j, i]

            if left[j, i]:
                # linke Seite: unten -> oben
                s = (y - y0)
            elif top[j, i]:
                # obere Seite: links -> rechts
                s = h + (x - x0)
            elif right[j, i]:
                # rechte Seite: oben -> unten
                s = h + w + ((y0 + h) - y)
            else:  # bottom[j, i]
            # untere Seite: rechts -> links
                s = h + w + h + ((x0 + w) - x)

            cp_val = (p[j, i] - p_inf) / (0.5 * rho * U**2)
            Cp_punkte.append((s, cp_val))
    
            if mit_scherung:
                delta = max(min(dx, dy), 1e-12)
                u_t = u[j, i]*(-ny) + v[j, i]*(nx)
                tau_t = mu * (u_t / delta)
                Fx_tau -= tau_t * (-ny) * ds

        Aref = h

    F_D = Fx_p + Fx_tau
    cD  = F_D / (0.5 * rho * U**2 * (Aref + 1e-30))
    return {"Cd": float(cD), "F_D": float(F_D), "Fx_p": Fx_p, "Fx_tau": Fx_tau,
            "Cp_points": sorted(Cp_punkte, key=lambda x: x[0]), "p_inf": p_inf}

# SIMPLE-Hauptschleife

def simple(cfg, verbose=True):
    x_c, y_c, Xc, Yc, x_w, y_w, dx, dy = gitter(cfg["Lx"], cfg["Ly"], cfg["Nx"], cfg["Ny"])
    Nx, Ny = cfg["Nx"], cfg["Ny"]
    u, v, p = felder(Nx, Ny)
    fluid = fluid_maske(Xc, Yc, cfg["obst"])

    # Startfeld
    u[:] = cfg["U_in"]; v[:] = 0.0
    u, v = randbedingung_uv(u, v, cfg, fluid)
    p = randbedingung_p(p)

    # Historie
    hist = {
        "res_u_abs": [], "res_v_abs": [], "res_cont_abs": [],
        "res_u_norm": [], "res_v_norm": [], "res_cont_norm": [],
        "Cd": []
    }

    for outer in range(cfg["n_outer"]):
        # --- Momentum ---
        mom_u, mom_v = momentum(u, v, p, cfg["rho"], cfg["mu"], dx, dy, fluid)
        aEu, aWu, aNu, aSu, aPu, b_u = mom_u
        aEv, aWv, aNv, aSv, aPv, b_v = mom_v

        # Unterrelaxation
        u_old = u.copy(); v_old = v.copy()
        alpha_u = max(1e-6, cfg["alpha_u"])
        aPu_mod = aPu / alpha_u
        b_u_mod = b_u + ((1.0 - alpha_u) / alpha_u) * aPu * u_old
        aPv_mod = aPv / alpha_u
        b_v_mod = b_v + ((1.0 - alpha_u) / alpha_u) * aPv * v_old

        # Loesen u*, v*
        res_u_eq, _ = gauss_seidel(u, aEu, aWu, aNu, aSu, aPu_mod, b_u_mod, fluid, itmax=140, tol=1e-8)
        res_v_eq, _ = gauss_seidel(v, aEv, aWv, aNv, aSv, aPv_mod, b_v_mod, fluid, itmax=140, tol=1e-8)

        # Konservativ
        u[~np.isfinite(u)] = 0.0; v[~np.isfinite(v)] = 0.0
        u[:] = np.clip(u, -10.0*cfg["U_in"], 10.0*cfg["U_in"])
        v[:] = np.clip(v, -10.0*cfg["U_in"], 10.0*cfg["U_in"])

        # Flussdivergenz (Kontinuituet)
        Fe, Fw, Fn, Fs = fluesse(u, v, cfg, dx, dy) # diskrete Form der Gleichung 2.1.2 
        div_mass = Fe - Fw + Fn - Fs  # kg/s
        res_cont_abs = float(np.sqrt(np.mean(np.nan_to_num(div_mass)**2))) #L2-Norm

        # Druckkorrektur (face-basiert)
        vol = dx * dy
        safe_aPu = aPu_mod.copy(); safe_aPu[~fluid] = 1.0
        safe_aPv = aPv_mod.copy(); safe_aPv[~fluid] = 1.0
        d_u = vol / (safe_aPu + 1e-30)
        d_v = vol / (safe_aPv + 1e-30)
        d_u_e = 0.5*(d_u[:, 0:-1] + d_u[:, 1:])
        d_v_n = 0.5*(d_v[0:-1, :] + d_v[1:, :])

        aEp = np.zeros_like(p); aWp = np.zeros_like(p)
        aNp = np.zeros_like(p); aSp = np.zeros_like(p)
        aPp = np.zeros_like(p); b_p = np.zeros_like(p)

        aEp[:, 0:Nx-1] = cfg["rho"] * d_u_e * dy / dx
        aWp[:, 1:Nx]   = cfg["rho"] * d_u_e * dy / dx
        aNp[0:Ny-1, :] = cfg["rho"] * d_v_n * dx / dy
        aSp[1:Ny,   :] = cfg["rho"] * d_v_n * dx / dy
        aPp = aEp + aWp + aNp + aSp + 1e-30

        b_p = (Fe - Fw + Fn - Fs)
        aPp[0, 0] = 1.0; b_p[0, 0] = 0.0

        p_prime = np.zeros_like(p)
        res_p_eq, _ = gauss_seidel(p_prime, aEp, aWp, aNp, aSp, aPp, -b_p, np.ones_like(fluid, dtype=bool), itmax=400, tol=1e-8)

        # Korrektur
        p = p + cfg["alpha_p"] * p_prime
        for j in range(1, Ny-1):
            for i in range(1, Nx-1):
                if not fluid[j, i]:
                    u[j, i] = 0.0; v[j, i] = 0.0
                    continue
                dpdx_p = (p_prime[j, i+1] - p_prime[j, i-1])/(2*dx)
                dpdy_p = (p_prime[j+1, i] - p_prime[j-1, i])/(2*dy)
                u[j, i] += - d_u[j, i] * (dpdx_p if np.isfinite(dpdx_p) else 0.0)
                v[j, i] += - d_v[j, i] * (dpdy_p if np.isfinite(dpdy_p) else 0.0)

        # BCs erneut
        u[~np.isfinite(u)] = 0.0; v[~np.isfinite(v)] = 0.0
        u[:] = np.clip(u, -10.0*cfg["U_in"], 10.0*cfg["U_in"])
        v[:] = np.clip(v, -10.0*cfg["U_in"], 10.0*cfg["U_in"])
        u, v = randbedingung_uv(u, v, cfg, fluid)
        p = randbedingung_p(p)

        # --- Residuen ---
        #Aenderungsnormen (L2) zwischen Outer-Iterationen
        res_u_abs = float(np.sqrt(np.mean((u - u_old)**2)))
        res_v_abs = float(np.sqrt(np.mean((v - v_old)**2)))
        # Normierungen
        norm_u = float(np.sqrt(np.mean(u_old**2))) + 1e-30
        norm_v = float(np.sqrt(np.mean(v_old**2))) + 1e-30
        res_u_norm = res_u_abs / norm_u
        res_v_norm = res_v_abs / norm_v
        res_cont_norm = res_cont_abs / (cfg["M_in"] + 1e-30)

        hist["res_u_abs"].append(res_u_abs)
        hist["res_v_abs"].append(res_v_abs)
        hist["res_cont_abs"].append(res_cont_abs)
        hist["res_u_norm"].append(res_u_norm)
        hist["res_v_norm"].append(res_v_norm)
        hist["res_cont_norm"].append(res_cont_norm)

        # --- c_D pro Iteration ---
        kraft_it = kraefte_und_cd(u, v, p, cfg, (x_c, y_c, Xc, Yc, dx, dy), fluid, n_theta=360, mit_scherung=True)
        hist["Cd"].append(kraft_it["Cd"])

        max_res = max(res_u_eq, res_v_eq, res_p_eq)
        if verbose and outer % 20 == 0:
            print(f"Outer {outer+1}/{cfg['n_outer']}: eq_res={max_res:.2e}, cont={res_cont_abs:.2e}, Cd~{kraft_it['Cd']:.3f}")
        # Konvergenz: Gleichungsrest + Kontinuitaet
        if (max_res < cfg["tol"]) and (res_cont_abs < 1e-6):
            if verbose:
                print(f"Konvergenz nach {outer+1} Outer-Iterationen erreicht.")
            break

    grid_roh = (x_c,y_c, Xc, Yc, x_w, y_w)
    grid_solver = (x_c, y_c, Xc, Yc, dx, dy)
    feld = (u, v, p)
    return grid_roh, grid_solver, feld, fluid, hist, (dx, dy)



# =========================
# 3) Post processing: Ergebnisse/Plots
# =========================
def plot_cp(forces, cfg):
    pts = forces.get("Cp_points", [])
    if not pts:
        print("Keine Cp-Punkte vorhanden.")
        return

    plt.figure(figsize=(6.3, 3.2))

    if cfg["obst"][0] == "circle":
        #(theta [rad], Cp)
        theta = np.array([t for t, cp in pts])
        cp = np.array([cp for t, cp in pts])
        theta_deg = theta * 180.0 / np.pi

        # sortieren nach Winkel 
        idx = np.argsort(theta_deg)
        theta_deg = theta_deg[idx]
        cp = cp[idx]

        plt.plot(theta_deg, cp, color='black', linewidth=1.5)
        plt.xlabel(r'$\theta$ [deg]')
        plt.ylabel(r'$C_P$ [-]')
        plt.title(r'Druckbeiwertverteilung $C_P(\theta)$ Kreis')
        plt.grid(True, alpha=0.3)

    else:
        #(s [m] entlang Umfang, Cp)
        s = np.array([s for s, cp in pts])
        cp = np.array([cp for s, cp in pts])

        # sortieren nach s
        idx = np.argsort(s)
        s = s[idx]
        cp = cp[idx]

        plt.plot(s, cp, color='black', linewidth=1.5)
        plt.xlabel(r'$s$ [m]')
        plt.ylabel(r'$C_P$ [-]')
        plt.title(r'Druckbeiwertverteilung $C_P(s)$ Rechteck')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def visualisierungen(grid, feld, hist, cfg, fluid, streamline_density=2.0):
    x_c, y_c, Xc, Yc, x_w, y_w = grid
    u, v, p = feld
    X, Y = np.meshgrid(x_c, y_c)
    speed = np.sqrt(u**2 + v**2)

    # 1) Druckkonturen
    plt.figure(figsize=(7.5, 2.4))
    cs = plt.contourf(X, Y, p, levels=40, cmap='jet')
    cbar = plt.colorbar(cs); cbar.set_label('[Pa]')
    zeichne_hindernis(plt.gca(), cfg["obst"])
    plt.title('Druck [Pa]'); plt.xlabel('x'); plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box'); plt.tight_layout(); plt.show()

    # 2) Stromlinien
    plt.figure(figsize=(7.5, 2.4))
    strm = plt.streamplot(X, Y, u, v, color=speed, cmap='jet', density=streamline_density, linewidth=1.0, arrowsize=0.9)
    cbar = plt.colorbar(strm.lines); cbar.set_label('[m/s]')
    zeichne_hindernis(plt.gca(), cfg["obst"])
    plt.title('Geschwindigkeit-Streamlines'); plt.xlabel('x'); plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box'); plt.tight_layout(); plt.show()
    
    # 3) Vektoren mit Hintergrund |v|
    plt.figure(figsize=(7.5, 2.4))
    plt.contourf(X, Y, speed, levels=40, cmap='jet')
    cbar = plt.colorbar(); cbar.set_label('[m/s]')
    skipx = max(1, len(x_c)//60); skipy = max(1, len(y_c)//30)
    plt.quiver(X[::skipy, ::skipx], Y[::skipy, ::skipx],
               u[::skipy, ::skipx], v[::skipy, ::skipx], color='k', scale=20)
    zeichne_hindernis(plt.gca(), cfg["obst"])
    plt.title('Geschwindigkeit-Vektoren'); plt.xlabel('x'); plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box'); plt.tight_layout(); plt.show()

    # 4) Residuen (normiert)
    it = np.arange(1, len(hist.get("res_u_norm", [])) + 1)
    if len(it) > 0:
        plt.figure(figsize=(7.2, 3.2))
        plt.semilogy(it, hist["res_cont_norm"], color='#3dd6c5', label='Kontinuität')
        plt.semilogy(it, hist["res_u_norm"],    color='#9b8fe3', label='x-Geschwindigkeit')
        plt.semilogy(it, hist["res_v_norm"],    color='#e34234', label='y-Geschwindigkeit')
        plt.legend()
        plt.title('Residuen über Iterationen')
        plt.xlabel('Iteration'); plt.ylabel('Residuen [-]')
        plt.grid(True, which='both', alpha=0.25)
        plt.tight_layout(); plt.show()

    # 5) c_D Verlauf
    if hist.get("Cd"):
        plt.figure(figsize=(7.2, 3.0))
        it_cd = np.arange(1, len(hist["Cd"]) + 1)
        plt.plot(it_cd, hist["Cd"], color='black', linewidth=1.6)
        plt.title('Widerstandsbeiwert $C_D$ über Iterationen')
        plt.xlabel('Iteration'); plt.ylabel(r'$C_D$ [-]')
        plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()


#Main

def main():
    cfg = eingaben()
    grid_roh, grid_solver, feld, fluid, hist, steps = simple(cfg, verbose=True)
    # Netz anzeigen
    plot_mesh(grid_roh[4], grid_roh[5], cfg["obst"])
    # Kraefte
    forces = kraefte_und_cd(feld[0], feld[1], feld[2], cfg, grid_solver, fluid, n_theta=720, mit_scherung=True)
    plot_cp(forces, cfg)
    print(f"Re = {cfg['Re']:.1f}, mu = {cfg['mu']:.3e} Pas (aus Re)")
    print(f"c_D â{forces['Cd']:.4f} (F_D â {forces['F_D']:.3e} N/m)")
    geom = cfg['obst'][0]
    print(f"Fx_druck = {forces['Fx_p']:.3e}, Fx_scherung = {forces['Fx_tau']:.3e}, geom: {geom}")

    # ANSYS-like Plots
    visualisierungen(grid_roh, feld, hist, cfg, fluid, streamline_density=2.0)

if __name__ == "__main__":
    main()
