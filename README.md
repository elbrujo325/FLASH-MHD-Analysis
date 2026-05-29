# FLASH-MHD Analysis Toolkit

## 📊 Overview

A comprehensive analysis suite for MHD accretion disk simulations using the **FLASH code** (University of Rochester). This toolkit processes HDF5 checkpoint files to compute key physical quantities including disk height (H), kurtosis, probability density profiles P(z), and energy evolution (kinetic, magnetic, thermal, gravitational).

Designed for analyzing magnetized accretion disks around compact objects, with specific applications to black hole systems.

## 🔬 Features

- **Disk Height Calculation**: Computes H(t) as density-weighted standard deviation of vertical coordinate
- **Kurtosis Analysis**: Measures deviation from Gaussian distribution in P(z)
- **Probability Density Evolution**: Generates animated GIFs of P(z) evolution over time
- **Energy Budget Tracking**: Tracks kinetic, magnetic, thermal, and gravitational energy evolution
- **Automated Visualization**: Produces publication-quality plots and animations
- **Fortran Par File Parsing**: Extracts simulation parameters (BETA, gamma, etc.) from flash.par

## 📁 Repository Structure

```
FLASH-MHD-Analysis/
├── grafica_HvsT---adc3727e-71bc-42a9-a020-2f0aedffe2df.py  # Main analysis script
└── README.md                                                # This file
```

## 🚀 Usage

### Prerequisites

```bash
# Core dependencies
pip install h5py numpy matplotlib scipy imageio

# Optional for GIF generation
pip install imageio[ffmpeg]
```

### Running the Analysis

1. **Modify paths**: Update `carpeta` variable in the script to point to your FLASH simulation directory
   ```python
   carpeta = "/path/to/your/FLASH/simulation/data"
   ```

2. **Execute the script**:
   ```bash
   python grafica_HvsT---adc3727e-71bc-42a9-a020-2f0aedffe2df.py
   ```

### Outputs Generated

- `H_y_kurtosis_paneles_beta{value}.png` - Side-by-side H(t) and Kurtosis(t) plots
- `evolucion_probabilidad_beta{value}.gif` - Animated P(z) evolution
- `energias_cuadricula_J.png` - 5-panel energy evolution grid (kinetic, magnetic, thermal, gravitational, total)

## 🧠 Physics Background

This toolkit analyzes magnetized accretion disks simulated with the FLASH code using:

- **Gravitational Potential**: Paczynski-Wiita pseudo-Newtonian potential
- **MHD Equations**: Ideal magnetohydrodynamics with adiabatic index γ
- **Coordinate System**: Cylindrical (R,Z) with symmetry in φ
- **Units**: Code units where G = M = R_S = 1 (typical for black hole simulations)

### Key Quantities Computed

**Disk Height (H)**:
```
H = √[⟨z²⟩ - ⟨z⟩²]
```
where averages are density-weighted over the radial range [rmin, rmax].

**Kurtosis (K)**:
```
K = ⟨(z - ⟨z⟩)⁴⟩ / ⟨(z - ⟨z⟩)²⟩²
```
Measures "tailedness" of P(z); K=3 for Gaussian distribution.

**Energy Components**:
- Kinetic: E_kin = ½∫ρv² dV
- Magnetic: E_mag = ∫(B²/8π) dV  
- Thermal: E_th = ∫P/(γ-1) dV
- Gravitational: E_grav = ∫ρΦ dV

## 📈 Sample Outputs

![Disk Height and Kurtosis](H_y_kurtosis_paneles_beta*.png)
*Left: Disk height evolution | Right: Kurtosis evolution (dashed line = Gaussian)*

![Probability Density Evolution](evolucion_probabilidad_beta*.gif)
*Animated evolution of P(z) with theoretical Gaussian overlay*

![Energy Evolution Grid](energias_cuadricula_J.png)
*Five-panel view of all energy components plus total*

## 🛠️ Customization

### Analysis Parameters
Modify these variables in the script:
- `rmin`, `rmax`: Radial range for disk analysis (default: 3.0 to 4.7)
- `var`: Variable to analyze (default: "dens" for density)
- `nbins`: Histogram bins for P(z) calculation (default: 200)

### Plot Styling
Adjust colors, line widths, and fonts in the matplotlib sections:
- Color scheme: royalblue, darkorange, navy, darkred
- Figure sizes and DPI settings

## 🔭 Scientific Applications

This analysis is particularly useful for:
- Studying MRI-driven turbulence in accretion disks
- Quantifying disk thickness evolution
- Analyzing energy dissipation mechanisms
- Comparing simulation results with theoretical models
- Validating MHD codes against analytical solutions

## 👨‍🔬 Author

**Paolo Alfaro Sotil**  
Physicist · UNMSM (San Marcos, Lima)  
Specializing in computational physics and MHD simulations

## 📄 License

MIT License - feel free to use and adapt for your research

## 🙏 Acknowledgements

- FLASH code developers (University of Rochester)
- UNMSM Physics Department
- Open-source scientific Python ecosystem (h5py, numpy, matplotlib, scipy)