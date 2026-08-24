# 🧬 Weak Interaction Barcode (WIB) Toolkit

**WIB (Weak Interaction Barcode)** is an automated software toolkit designed for the dynamic barcoding and dimensionality reduction of non-covalent interactions, including hydrogen bonds and π-π stacking, in molecular dynamics simulations. It seamlessly integrates native quantitative analysis into a unified framework, providing actionable mechanistic insights for biomolecular research.

> **🌟 Important Notice:** 
> For detailed usage documentation, step-by-step tutorials, and complete example datasets covering various biological systems (e.g., protein-protein, protein-nucleic acid, aptamer-small molecule), please refer to our comprehensive guides hosted on Zenodo: **[https://zenodo.org/records/22007579](https://zenodo.org/records/22007579)**

> **📝 Citation:** 
> If you use this tool for hydrogen bond and π-π analysis in your research system, please be sure to cite our work. Thank you!

---

## 💻 1. Installation & Requirements

WIB is a cross-platform Python analytical framework tightly integrated with GROMACS. 

**📌 Prerequisites:**
* 🐍 Python >= 3.8
* 💧 GROMACS >= 2024 (recommended)
* 🐧 Operating System: Linux or macOS (recommended)

**📦 Installation via pip:**

```bash
# Method 1: Direct installation from GitHub (Recommended)
pip install git+https://github.com/Jackshoulder/WIB.git

# Method 2: Local build
git clone https://github.com/Jackshoulder/WIB.git
cd WIB
pip install .
```

✅ **Verify the installation** by accessing the built-in help commands:

```bash
wib-hbond -h
wib-pipi -h
```

---

## 🗺️ 2. General Workflow

The standard execution of WIB relies on highly readable YAML configuration files (`hb_config.yaml` for hydrogen bonds and `pipi_config.yaml` for π-π stacking). 

> 💡 **Before formal operation:** Copy the core configuration files into your working directory and verify that the file paths for the trajectory file (`.xtc`), topology file (`.tpr`), and output directory (`outdir`) have been correctly specified.

### 🔍 Step 1: Topology Inspection (Highly Recommended)

Before conducting large-scale global searches, it is highly recommended to enable the topology reconnaissance mode by using the `-inspect` (or `-i`) parameter. This step provides accurate grouping and pairing suggestions for your specific system.

```bash
# For Hydrogen Bond analysis
wib-hbond hb_config.yaml -i

# For pi-pi stacking analysis
wib-pipi pipi_config.yaml -i
```

**📊 Terminal Feedback and Configuration Parsing:**
* 🧬 **Topology Verification:** The system will parse and print specific molecular information from the TPR file, including actual chain IDs, residue number ranges, and global absolute atom numbers.
* 🏷️ **Macro Group (Groups) Recommendation:** The system recommends writing formats for the absolute atomic numbers assigned to the receptor (Group 1) and ligand (Group 2). Users should carefully check these numbers and complete the accurate assignments in the YAML file.
* 🎯 **Target Matrix (Pairs) Recommendation:** 
    * 🔹 **Exhaustive Mode:** Recommended for exploring unknown binding interfaces (wide-range enumeration). The terminal lists all parsed identifiers, which users classify into `list_A` and `list_B` based on their research objectives.
    * 🔹 **Explicit Mode:** Recommended for analyzing specific, known interactions, such as intramolecular base pairings in an aptamer stem. Users explicitly define interacting pairs in an `explicit_list`.
* 💍 **Ring Setting (For π-π Analysis):** The system checks for non-standard ligand rings. If detected, it prints the corresponding ring information, which must be copied into the `rings` section of the YAML file. Default geometric criteria are a centroid distance cutoff of 0.45 nm and an offset angle cutoff of 30.0°.

### ⚙️ Step 2: Plot Configuration

After defining the targets, customize the plot drawing control block in your YAML file according to your accuracy requirements:

* 📉 **`occ_threshold`**: Draw high-resolution barcodes only for residue pairs with an occupancy greater than this value (e.g., `10.0` or `50.0`).
* 📈 **`kde_threshold`**: Generate separate distance-angle conformational matrix analysis plots only for "elite residue pairs" with an occupancy rate greater than this specified value.
* 🧹 **`clean_threshold`**: Automatic garbage collection removes underlying temporary files occupying less than this threshold, significantly freeing up hard disk space.

*✨ Tip: If you find an excessive or insufficient number of interactions, appropriately decrease or increase `occ_threshold` and `kde_threshold`.*

### 🚀 Step 3: Automated Execution

Start the official calculation process:

```bash
wib-hbond hb_config.yaml
# or
wib-pipi pipi_config.yaml
```

**🔬 Operation and Analysis Process:**
1. **⚡ Rapid Pre-Screening:** The system first performs a distance-based global preliminary screening. Residue pairs (or ring combinations) that severely deviate from geometric standards are directly eliminated, instantly removing a vast majority of spatially invalid pairings and significantly reducing computational overhead.
2. **🧮 Precise Calculation:** The program conducts precise geometric and occupancy calculations on the retained candidate pairs.
3. **📁 Data Output & Visualization:** The terminal intuitively prints core information (occupancy rate, average distance, average angle). Automatically generated dynamic barcode panel plots and standardized CSV datasets (containing detailed occupancy rates and time series) are saved in the specified `outdir` for secondary plotting and in-depth interpretation.

### ⚛️ Step 4: Atomic-Level Mapping (H-Bonds Only)

After decomposition via residue-level hydrogen bond barcodes, you can further extract highly specific atomic-level hydrogen bond maps.

Use a separate configuration file (e.g., `hb_config-atoms.yaml`) with the output directory properly configured to avoid overwriting previous data.

```bash
wib-hbond hb_config-atoms.yaml -a
```
