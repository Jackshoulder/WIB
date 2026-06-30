#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=========================================================================================
GROMACS Pi-Pi Stacking Automated Analysis & Intelligent Inspector 
Sub-molecular Pi-Pi Stacking Analysis & Smart Small Molecule Graph Inspector (Absolute Coordinate Atom Mapping Edition V26)
=========================================================================================
"""

import os
import sys
import shutil
import subprocess
import logging
import argparse
import itertools
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import concurrent.futures
import multiprocessing
from tqdm import tqdm

class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# Core Design: Standard Molecular Library and Classification Tags
# ==========================================
BUILTIN_RES_TYPES = {
    'A':   {'6R': "N1 C2 N3 C4 C5 C6", '5R': "C4 C5 N7 C8 N9", 'P6': "N1 N3 C5", 'P5': "C4 N7 N9"},
    'ADE': {'6R': "N1 C2 N3 C4 C5 C6", '5R': "C4 C5 N7 C8 N9", 'P6': "N1 N3 C5", 'P5': "C4 N7 N9"},
    'DA':  {'6R': "N1 C2 N3 C4 C5 C6", '5R': "C4 C5 N7 C8 N9", 'P6': "N1 N3 C5", 'P5': "C4 N7 N9"},
    'G':   {'6R': "N1 C2 N3 C4 C5 C6", '5R': "C4 C5 N7 C8 N9", 'P6': "N1 N3 C5", 'P5': "C4 N7 N9"},
    'GUA': {'6R': "N1 C2 N3 C4 C5 C6", '5R': "C4 C5 N7 C8 N9", 'P6': "N1 N3 C5", 'P5': "C4 N7 N9"},
    'DG':  {'6R': "N1 C2 N3 C4 C5 C6", '5R': "C4 C5 N7 C8 N9", 'P6': "N1 N3 C5", 'P5': "C4 N7 N9"},
    'C':   {'6R': "N1 C2 N3 C4 C5 C6", 'P6': "N1 N3 C5"},
    'CYT': {'6R': "N1 C2 N3 C4 C5 C6", 'P6': "N1 N3 C5"},
    'DC':  {'6R': "N1 C2 N3 C4 C5 C6", 'P6': "N1 N3 C5"},
    'U':   {'6R': "N1 C2 N3 C4 C5 C6", 'P6': "N1 N3 C5"},
    'URA': {'6R': "N1 C2 N3 C4 C5 C6", 'P6': "N1 N3 C5"},
    'RU':  {'6R': "N1 C2 N3 C4 C5 C6", 'P6': "N1 N3 C5"},
    'T':   {'6R': "N1 C2 N3 C4 C5 C6", 'P6': "N1 N3 C5"},
    'THY': {'6R': "N1 C2 N3 C4 C5 C6", 'P6': "N1 N3 C5"},
    'DT':  {'6R': "N1 C2 N3 C4 C5 C6", 'P6': "N1 N3 C5"},
    'PHE': {'6R': "CG CD1 CD2 CE1 CE2 CZ", 'P6': "CG CE1 CE2"},
    'TYR': {'6R': "CG CD1 CD2 CE1 CE2 CZ", 'P6': "CG CE1 CE2"},
    'TRP': {'6R': "CD2 CE2 CE3 CZ3 CH2 CZ2", '5R': "CG CD1 NE1 CE2 CD2", 'P6': "CD2 CE3 CH2", 'P5': "CG NE1 CD2"},
    'HIS': {'5R': "CG ND1 CE1 NE2 CD2", 'P5': "CG CE1 CD2"},
    'HSD': {'5R': "CG ND1 CE1 NE2 CD2", 'P5': "CG CE1 CD2"},
    'HSE': {'5R': "CG ND1 CE1 NE2 CD2", 'P5': "CG CE1 CD2"},
    'HSP': {'5R': "CG ND1 CE1 NE2 CD2", 'P5': "CG CE1 CD2"}
}

NA_RESNAMES = {"A", "ADE", "DA", "G", "GUA", "DG", "C", "CYT", "DC", "U", "URA", "RU", "T", "THY", "DT"}
PROT_RESNAMES = {"PHE", "TYR", "TRP", "HIS", "HSD", "HSE", "HSP"}
IGNORE_RESNAMES = {
    "SOL", "WAT", "HOH", "NA", "CL", "K", "MG", "CA", "ZN", "ION", "TIP3P",
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "ILE", "LEU", "LYS", "MET", "PRO", "SER", "THR", "VAL"
}

def setup_logger(output_dir):
    log_file = os.path.join(output_dir, "pipi_analysis_process.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                                  logging.StreamHandler(sys.stdout)])
    return log_file

def parse_yaml_config(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# ==========================================
# Smart Structure Parsing Engine: Forcing Chain ID Tags to Prevent Duplication
# ==========================================
def analyze_structure_rings(pdb_path):
    residues_data = {}
    category_rings = {'NA': {}, 'PROT': {}, 'MOL': {}}
    seen_builtins = set()
    
    if not os.path.exists(pdb_path): return category_rings, {}

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    serial = int(line[6:11].strip())
                    atomname = line[12:16].strip()
                    resname = line[17:21].strip().upper()
                    chain = line[21].strip() or "X"
                    resnr = int(line[22:26].strip())
                    x, y, z = float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())
                    element = atomname[0]
                    
                    if resname in IGNORE_RESNAMES: continue
                        
                    if resname in BUILTIN_RES_TYPES:
                        builtin_key = f"{chain}_{resname}{resnr}"
                        if builtin_key not in seen_builtins:
                            seen_builtins.add(builtin_key)
                            for ring_type in BUILTIN_RES_TYPES[resname].keys():
                                if ring_type.endswith('R'):
                                    tag = f"{chain}_{resname}{resnr}_{ring_type}"
                                    if resname in NA_RESNAMES:
                                        if chain not in category_rings['NA']: category_rings['NA'][chain] = []
                                        category_rings['NA'][chain].append(tag)
                                    else:
                                        if chain not in category_rings['PROT']: category_rings['PROT'][chain] = []
                                        category_rings['PROT'][chain].append(tag)
                        continue
                        
                    res_key = f"{chain}_{resname}_{resnr}"
                    if res_key not in residues_data: residues_data[res_key] = []
                    if element in ['C', 'N', 'O', 'S', 'P']:
                        residues_data[res_key].append({'serial': serial, 'name': atomname, 'pos': np.array([x, y, z])})
                except: pass

    ligand_defs = {}
    for res_key, atoms in residues_data.items():
        n = len(atoms)
        if n < 5: continue
        
        adj = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(atoms[i]['pos'] - atoms[j]['pos'])
                if 1.15 <= dist <= 1.75:
                    adj[i].append(j)
                    adj[j].append(i)
        
        cycles = []
        def dfs(node, start, path):
            if len(path) in [5, 6]:
                if start in adj[node]:
                    c = sorted(path)
                    if c not in cycles: cycles.append(c)
            if len(path) >= 6: return
            for nxt in adj[node]:
                if nxt not in path:
                    if len(path) > 1 and nxt == start: continue
                    dfs(nxt, start, path + [nxt])

        for start in range(n):
            dfs(start, start, [start])

        if cycles:
            ligand_defs[res_key] = []
            for idx, c in enumerate(cycles):
                ring_atoms = [atoms[idx]['name'] for idx in c]
                if len(c) == 6: plane_atoms = [atoms[c[0]]['name'], atoms[c[2]]['name'], atoms[c[4]]['name']]
                else: plane_atoms = [atoms[c[0]]['name'], atoms[c[2]]['name'], atoms[c[3]]['name']]
                
                chain, resname_raw, resnr_raw = res_key.split('_')
                tag = f"{chain}_{resname_raw}{resnr_raw}_{len(c)}R"
                if len(cycles) > 1: tag += f"_{idx+1}"
                
                if chain not in category_rings['MOL']: category_rings['MOL'][chain] = []
                category_rings['MOL'][chain].append(tag)
                
                ligand_defs[res_key].append({
                    'tag': tag, 'chain': chain, 'resnr': resnr_raw, 'com': " ".join(ring_atoms), 'plane': " ".join(plane_atoms)
                })
                
    return category_rings, ligand_defs

def inspect_topology_mode(gmx_cmd, tpr, traj, outdir):
    os.makedirs(outdir, exist_ok=True)
    print(f"\n{Colors.CYAN}{'='*80}\n GROMACS Pi-Pi Topology Inspector \n{'='*80}{Colors.RESET}")
    ref_pdb = os.path.join(outdir, "pipi_inspect_ref.pdb")
    
    print(f"{Colors.YELLOW}[*] Extracting spatial reference PDB from TPR...{Colors.RESET}")
    subprocess.run([gmx_cmd, "editconf", "-f", tpr, "-o", ref_pdb], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"{Colors.YELLOW}[*] Initializing intelligent multi-chain classification engine...{Colors.RESET}")
    category_rings, ligand_defs = analyze_structure_rings(ref_pdb)
    
    def get_all(cat_dict): return [tag for tags in cat_dict.values() for tag in tags]
    def format_chain_info(cat_dict): return ", ".join([f"Chain {c if c != 'X' else 'Unassigned'}: {len(tags)}" for c, tags in cat_dict.items()])

    na_all, prot_all, mol_all = get_all(category_rings['NA']), get_all(category_rings['PROT']), get_all(category_rings['MOL'])
    
    print(f"\n{Colors.GREEN}[ 1. Smart Diagnosis of System Molecule Types ]:{Colors.RESET}")
    if na_all: print(f" {Colors.CYAN}--> 🧬 Found Nucleic Acid Aromatic Rings: {len(na_all)} ({format_chain_info(category_rings['NA'])}){Colors.RESET}")
    if prot_all: print(f" {Colors.CYAN}--> 🥩 Found Protein Aromatic Rings: {len(prot_all)} ({format_chain_info(category_rings['PROT'])}){Colors.RESET}")
    if mol_all: print(f" {Colors.CYAN}--> 💊 Found Ligand Aromatic Rings: {len(mol_all)} ({format_chain_info(category_rings['MOL'])}){Colors.RESET}")

    list_a_demo, list_b_demo = [], []
    all_chains = list(set(list(category_rings['NA'].keys()) + list(category_rings['PROT'].keys()) + list(category_rings['MOL'].keys())))
    
    if len(all_chains) >= 2:
        demo_chain_a, demo_chain_b = all_chains[0], all_chains[1]
        print(f"\n{Colors.YELLOW} 💡 Detected multiple chains in the system! [Overlap Protection] enabled.{Colors.RESET}")
        print(f"{Colors.YELLOW}    All aromatic rings are forced to bind with chain prefixes (e.g., {demo_chain_a}_TYR32_6R).{Colors.RESET}")
        print(f"{Colors.YELLOW}    Please select the rings you need from the [Candidate Ring Dictionary] below and place them into list_A and list_B.{Colors.RESET}")
        
        if demo_chain_a in category_rings['PROT']: list_a_demo = category_rings['PROT'][demo_chain_a][:3]
        elif demo_chain_a in category_rings['NA']: list_a_demo = category_rings['NA'][demo_chain_a][:3]
        
        if demo_chain_b in category_rings['PROT']: list_b_demo = category_rings['PROT'][demo_chain_b][:3]
        elif demo_chain_b in category_rings['MOL']: list_b_demo = category_rings['MOL'][demo_chain_b][:3]
    else:
        list_a_demo, list_b_demo = (prot_all + na_all + mol_all)[:3], (prot_all + na_all + mol_all)[-3:]

    print(f"\n{Colors.GREEN}[ 2. YAML Configuration Template Generation (Copy & Paste) ]:{Colors.RESET}")
    print(f"{Colors.RED}# =========================================={Colors.RESET}")
    print(f"system:\n  traj: \"{traj}\"\n  tpr: \"{tpr}\"\n  outdir: \"{outdir}\"\n  force: true")
    print(f"\ncriteria:\n  dist_cutoff: 0.45\n  ang_cutoff: 30.0")
    
    print(f"\nrings:")
    if ligand_defs:
        for res, rings in ligand_defs.items():
            for r in rings:
                print(f"  {r['tag']}: {{chain: \"{r['chain']}\", resnr: {r['resnr']}, com_atoms: \"{r['com']}\", plane_atoms: \"{r['plane']}\"}}")
    else:
        print(f"  # The system contains no non-standard ligands; built-in standard aromatic residues can be used directly.")

    print(f"\npairs:\n  mode: \"exhaustive\"")
    print(f"  list_A:")
    print(f"    # --- Paste Receptor/Chain A aromatic rings here ---")
    for tag in list_a_demo: print(f"    - \"{tag}\"")
    print(f"  list_B:")
    print(f"    # --- Paste Ligand/Chain B aromatic rings here ---")
    for tag in list_b_demo: print(f"    - \"{tag}\"")
        
    print(f"\n  # ========================================================================")
    print(f"  # 🛡️ Candidate Ring Dictionary (Chain-Aware)")
    print(f"  # To avoid overlapping residue numbers across chains (e.g., TYR32 in Chain A and B), all rings are auto-prefixed.")
    print(f"  # Copy and assemble tags from the list below into list_A and list_B based on your research needs.")
    print(f"  # ========================================================================")
    
    for cat_name, cat_dict in [("Nucleic Acid", category_rings['NA']), ("Protein", category_rings['PROT']), ("Ligand (MOL)", category_rings['MOL'])]:
        for chain, tags in cat_dict.items():
            print(f"  # --- [ {cat_name} | Chain {chain} ] ---")
            for tag in tags:
                print(f"  # - \"{tag}\"")

    print(f"\nplot:\n  occ_threshold: 1.0\n  clean_threshold: 1.0\n  cbar_max: 1")
    print(f"{Colors.RED}# =========================================={Colors.RESET}\n")
    sys.exit(0)

# ==========================================
# Universal Definition Resolver (Extracts Chain ID and Residue Number)
# ==========================================
def resolve_ring_definition(name, custom_rings):
    if name in custom_rings: return custom_rings[name]
    try:
        parts = name.split('_')
        if len(parts) >= 3:
            chain = parts[0]
            type_tag = parts[-1] 
            prefix = "_".join(parts[1:-1]) 
        elif len(parts) == 2:
            chain = "X"
            prefix, type_tag = parts[0], parts[1]
        else:
            return None
            
        resname = "".join([c for c in prefix if c.isalpha()]).upper()
        resnr = int("".join([c for c in prefix if c.isdigit()]))
        
        if resname in BUILTIN_RES_TYPES and type_tag in BUILTIN_RES_TYPES[resname]:
            lib = BUILTIN_RES_TYPES[resname]
            return {'chain': chain, 'resnr': resnr, 'com_atoms': lib[type_tag], 'plane_atoms': lib['P6' if type_tag == '6R' else 'P5']}
    except: pass
    return None

def generate_target_pairs(pair_config, custom_rings):
    mode = pair_config.get('mode', 'explicit')
    raw_pairs = []
    
    if mode == 'exhaustive':
        raw_pairs = list(itertools.product(pair_config.get('list_A', []), pair_config.get('list_B', [])))
    elif mode == 'explicit':
        if 'explicit_list' in pair_config:
            raw_pairs = [tuple(p) for p in pair_config['explicit_list']]
            
    unique_pairs = set()
    final_pairs = []
    for r1, r2 in raw_pairs:
        if r1 == r2: continue
        pair_key = tuple(sorted([r1, r2]))
        if pair_key not in unique_pairs:
            unique_pairs.add(pair_key)
            final_pairs.append((r1, r2))
    return final_pairs

# ==========================================
# GROMACS Selection Compiler: Absolute Coordinate Atomic Mapping (Chain Amnesia Prevention)
# ==========================================
def build_gmx_sel(def_dict, is_com, atom_map):
    chain = def_dict.get('chain', 'X')
    resnr = def_dict['resnr']
    atom_names = def_dict['com_atoms'].split() if is_com else def_dict['plane_atoms'].split()
    
    serials = []
    for aname in atom_names:
        key = (chain, resnr, aname)
        if key in atom_map:
            serials.append(str(atom_map[key]))
            
    prefix = "com of " if is_com else ""
    
    # If perfectly mapped, output absolute atomnr directly, immune to chain overlap and parsing bugs.
    if len(serials) == len(atom_names):
        return f"{prefix}atomnr {' '.join(serials)}"
    else:
        # Rarely triggered fallback mechanism
        ch_str = f"chain {chain} and " if chain != 'X' else ""
        return f"{prefix}{ch_str}resnr {resnr} and name {' '.join(atom_names)}"

# ==========================================
# Phase 1: Global Batch Distance Prescreening
# ==========================================
def perform_fast_discovery(gmx_cmd, traj, tpr, base_pairs, custom_rings, outdir, atom_map):
    batch_size = 250 
    valid_pairs = []
    batches = [base_pairs[i : i + batch_size] for i in range(0, len(base_pairs), batch_size)]

    for batch_idx, batch_pairs in enumerate(tqdm(batches, desc="Phase 1 (Fast Prescreen)", ascii=False, ncols=80)):
        sel_file = os.path.join(outdir, f"phase1_sel_{batch_idx}.dat")
        out_xvg = os.path.join(outdir, f"phase1_dist_{batch_idx}.xvg")

        with open(sel_file, "w", newline="\n") as f:
            for r1, r2 in batch_pairs:
                def1 = resolve_ring_definition(r1, custom_rings)
                def2 = resolve_ring_definition(r2, custom_rings)
                # Inject absolute coordinate directives
                sel = f'{build_gmx_sel(def1, True, atom_map)} plus {build_gmx_sel(def2, True, atom_map)}'
                f.write(f"{sel};\n")

        cmd = [gmx_cmd, "distance", "-s", tpr, "-f", traj, "-sf", sel_file, "-oall", out_xvg]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        
        if res.returncode != 0:
            return None, f"GROMACS Phase 1 Crash: {res.stderr}"

        data = []
        if os.path.exists(out_xvg):
            with open(out_xvg, 'r') as f:
                for line in f:
                    if not line.startswith(('@', '#')):
                        parts = line.split()
                        if len(parts) > 1: data.append([float(x) for x in parts])
        data = np.array(data)

        if data.size > 0:
            for i, (r1, r2) in enumerate(batch_pairs):
                col_idx = i + 1 
                if col_idx < data.shape[1]:
                    min_dist = np.min(data[:, col_idx])
                    if min_dist <= 0.8:
                        valid_pairs.append((r1, r2))

        if os.path.exists(out_xvg): os.remove(out_xvg)
        if os.path.exists(sel_file): os.remove(sel_file)

    return valid_pairs, ""
    
# ==========================================
# Phase 2: Core Parallel Computing Engine
# ==========================================
def run_single_pipi(task):
    gmx_cmd, traj, tpr, r1, r2, def1, def2, out_dist, out_ang, force, atom_map = task
    if force:
        if os.path.exists(out_dist): os.remove(out_dist)
        if os.path.exists(out_ang): os.remove(out_ang)
    try:
        sel_dist = f'{build_gmx_sel(def1, True, atom_map)} plus {build_gmx_sel(def2, True, atom_map)}'
        if not os.path.exists(out_dist):
            res1 = subprocess.run([gmx_cmd, "distance", "-s", tpr, "-f", traj, "-select", sel_dist, "-oall", out_dist], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res1.returncode != 0: return False, f"{r1}-{r2}", f"Distance error: {res1.stderr}"
            
        sel_ang1 = f'{build_gmx_sel(def1, False, atom_map)}'
        sel_ang2 = f'{build_gmx_sel(def2, False, atom_map)}'
        if not os.path.exists(out_ang):
            res2 = subprocess.run([gmx_cmd, "gangle", "-s", tpr, "-f", traj, "-g1", "plane", "-g2", "plane", 
                                   "-group1", sel_ang1, "-group2", sel_ang2, "-oav", out_ang], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res2.returncode != 0: return False, f"{r1}-{r2}", f"Angle error: {res2.stderr}"
            
        return True, f"{r1}-{r2}", ""
    except Exception as e:
        return False, f"{r1}-{r2}", str(e)

def parse_xvg(xvg_file):
    if not os.path.exists(xvg_file): return np.array([]), np.array([])
    times, values = [], []
    with open(xvg_file, 'r') as f:
        for line in f:
            if not line.startswith(('@', '#')) and len(line.strip().split()) >= 2:
                parts = line.strip().split()
                times.append(float(parts[0]) / 1000.0) 
                values.append(float(parts[1]))
    return np.array(times), np.array(values)

def main():
    parser = argparse.ArgumentParser(description="Universal Intelligent Pi-Pi Stacking Core Analysis Engine")
    parser.add_argument("config", nargs="?", default="pipi_config.yaml", help="Path to config file")
    parser.add_argument("-i", "--inspect", action="store_true", help="Launch smart graph theory inspector mode to automatically mine small molecule ring structures")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"\n{Colors.YELLOW}Error: Config file not found!{Colors.RESET}\n")
        sys.exit(1)

    cfg = parse_yaml_config(args.config)
    sys_cfg, crit_cfg, plot_cfg = cfg['system'], cfg.get('criteria', {}), cfg.get('plot', {})
    custom_rings = cfg.get('rings', {}) or {}
    
    gmx_cmd, traj, tpr = sys_cfg.get('gmx_cmd', 'gmx'), sys_cfg.get('traj', 'fit.xtc'), sys_cfg.get('tpr', 'md_0_1.tpr')
    outdir, force = sys_cfg.get('outdir', 'pipi_out'), sys_cfg.get('force', False)

    if args.inspect: inspect_topology_mode(gmx_cmd, tpr, traj, outdir)

    dist_cutoff, ang_cutoff = float(crit_cfg.get('dist_cutoff', 0.45)), float(crit_cfg.get('ang_cutoff', 30.0))
    occ_threshold, clean_threshold = float(plot_cfg.get('occ_threshold', 1.0)), float(plot_cfg.get('clean_threshold', 1.0))

    if force and os.path.exists(outdir): shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    setup_logger(outdir)

    logging.info(f"=== Starting Universal YAML-Driven Pi-Pi Engine ===")
    
    # Core Pre-operation: Parse and build Absolute Coordinate Mapping (Atom Number Map)
    ref_pdb = os.path.join(outdir, "pipi_inspect_ref.pdb")
    if not os.path.exists(ref_pdb):
        logging.info(f"[*] Extracting spatial reference structure from TPR to establish underlying absolute coordinate mapping...")
        subprocess.run([gmx_cmd, "editconf", "-f", tpr, "-o", ref_pdb], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    atom_map = {}
    if os.path.exists(ref_pdb):
        with open(ref_pdb, 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    serial = int(line[6:11].strip())
                    atomname = line[12:16].strip()
                    chain = line[21].strip() or "X"
                    resnr = int(line[22:26].strip())
                    atom_map[(chain, resnr, atomname)] = serial
    
    base_pairs = generate_target_pairs(cfg['pairs'], custom_rings)
    
    logging.info(f"[*] Starting Phase 1: Global Trajectory Fast Prescreening (Total {len(base_pairs)} candidate pairs)...")
    valid_pairs, err_msg = perform_fast_discovery(gmx_cmd, traj, tpr, base_pairs, custom_rings, outdir, atom_map)
    
    if err_msg:
        logging.error(f"[!] Phase 1 Prescreening Failed: {err_msg}")
        sys.exit(1)
        
    reduced_count = len(base_pairs) - len(valid_pairs)
    logging.info(f"[+] Phase 1 Prescreening successful! Filtered out {reduced_count} invalid pairs, leaving only {len(valid_pairs)} core targets for precise calculation!")

    if not valid_pairs:
        logging.error("[!] No valid analysis tasks generated (All pairs are too far apart).")
        sys.exit(1)
    
    tasks = []
    for r1, r2 in valid_pairs:
        def1 = resolve_ring_definition(r1, custom_rings)
        def2 = resolve_ring_definition(r2, custom_rings)
        if def1 is None or def2 is None: continue
        out_dist = os.path.join(outdir, f"dist_{r1}_{r2}.xvg")
        out_ang = os.path.join(outdir, f"ang_{r1}_{r2}.xvg")
        # Mount atom_map
        tasks.append((gmx_cmd, traj, tpr, r1, r2, def1, def2, out_dist, out_ang, force, atom_map))

    cpu_cores = multiprocessing.cpu_count()
    max_threads = min(cpu_cores, 32)
    logging.info(f"[*] Starting Phase 2: High-performance parallel engine. Dispatching {len(tasks)} core calculation tasks to {max_threads} cores...")

    successful_targets = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        results = list(tqdm(executor.map(run_single_pipi, tasks), total=len(tasks), desc="Processing", ascii=False, ncols=80))
        for success, pair_name, msg in results:
            if success: successful_targets.append(pair_name)
            else: logging.error(f"  [X] Calculation Crash {pair_name}: {msg}")

    # ==========================================
    # Phase 3: Data Cleaning and Plotting (Publication-Standard Formatting)
    # ==========================================
    pair_data, valid_stats = {}, []
    times_total, pipi_total_timeline = None, None

    logging.info("\n=== Pi-Pi Stacking Occupancy and Bound State Geometric Feature Screening Results ===")
    for task in tasks:
        r1, r2 = task[3], task[4]
        pair_name = f"{r1}-{r2}"
        if pair_name not in successful_targets: continue
            
        t_dist, dists = parse_xvg(task[7])
        _, angles = parse_xvg(task[8])
        
        if len(dists) == 0 or len(dists) != len(angles): continue
        if times_total is None:
            times_total = t_dist
            pipi_total_timeline = np.zeros(len(times_total))
            
        total_frames = len(dists)
        status_array = np.zeros(total_frames)
        
        for i in range(total_frames):
            if (dists[i] <= dist_cutoff) and (angles[i] <= ang_cutoff or angles[i] >= 180 - ang_cutoff):
                status_array[i] = 1
                pipi_total_timeline[i] += 1
                
        occ = (np.sum(status_array) / total_frames) * 100
        
        if occ >= occ_threshold:
            pair_data[pair_name] = status_array
            formed_indices = np.where(status_array == 1)[0]
            avg_dist = np.mean(dists[formed_indices])
            formed_angles = angles[formed_indices]
            mapped_angles = np.where(formed_angles > 90, 180 - formed_angles, formed_angles)
            avg_ang = np.mean(mapped_angles)
            
            valid_stats.append({'Pair': pair_name, 'Occ': occ, 'Dist': avg_dist, 'Ang': avg_ang})
            logging.info(f" Target {pair_name:<25} | Occupancy: {occ:>6.2f}% | Centroid Dist: {avg_dist:>5.3f} nm | Deviation Ang: {avg_ang:>5.1f}°")

        if occ < clean_threshold:
            for fpath in [task[7], task[8]]:
                if os.path.exists(fpath): os.remove(fpath)

    if not pair_data:
        logging.error("[!] No pairs passed the final occupancy threshold filter.")
        sys.exit(1)

    pd.DataFrame(valid_stats).to_csv(os.path.join(outdir, "pipi_summary_stats.csv"), index=False, float_format='%.3f')
    barcode_matrix = np.array([pair_data[name] for name in pair_data.keys()])
    valid_occupancies = [(np.sum(pair_data[name]) / len(times_total)) * 100 for name in pair_data.keys()]
    
    df_barcode = pd.DataFrame(barcode_matrix.T, columns=list(pair_data.keys()))
    df_barcode.insert(0, 'Time_ns', times_total)
    df_barcode.to_csv(os.path.join(outdir, "pipi_barcode_data.csv"), index=False, float_format='%.3f')
    pd.DataFrame({'Time_ns': times_total, 'Total_PiPi': pipi_total_timeline}).to_csv(
        os.path.join(outdir, "pipi_total_timeline.csv"), index=False, float_format='%.3f')

    c_min_rgb = [x / 255.0 for x in plot_cfg.get('color_min', [255, 255, 255])]
    c_max_rgb = [x / 255.0 for x in plot_cfg.get('color_max', [147, 39, 143])]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_pipi_cmap", [c_min_rgb, c_max_rgb])
    
    num_pairs = len(pair_data)
    fig_height = max(10, num_pairs * 0.4 + 4)
    fig = plt.figure(figsize=(14, fig_height))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(times_total, pipi_total_timeline, color=tuple(c_max_rgb), linewidth=1.5, alpha=0.8)
    
    global_avg = np.mean(pipi_total_timeline)
    ax1.axhline(y=global_avg, color='gray', linestyle='--', linewidth=1.5)
    ax1.text(times_total[-1]*0.99, global_avg + 0.1, f' Avg: {global_avg:.2f} ', 
             color='black', va='bottom', ha='right', fontsize=12, fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', pad=2))
             
    ax1.set_title(r"Total $\pi$-$\pi$ Stacking Interactions over Time", fontsize=16, weight='bold', pad=10)
    ax1.set_ylabel(r"Instantaneous $\pi$-$\pi$ Count", fontsize=12)
    ax1.set_xlim(times_total[0], times_total[-1])
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = plt.subplot(2, 1, 2)
    sns.heatmap(barcode_matrix, cmap=custom_cmap, vmin=0, vmax=plot_cfg.get('cbar_max', 1), 
                cbar_kws={'label': 'State (1=Stacked, 0=Broken)'}, yticklabels=list(pair_data.keys()), ax=ax2)
                
    tick_positions = np.linspace(0, barcode_matrix.shape[1] - 1, 6)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"{times_total[int(idx)]:g}" for idx in tick_positions], rotation=0)
    
    ax2.set_title(rf"Dynamic $\pi$-$\pi$ Barcode Panel (Filtered > {occ_threshold}%)", fontsize=16, weight='bold', pad=10)
    ax2.set_xlabel("Time (ns)", fontsize=12)
    ax2.set_ylabel("Aromatic Ring Pairs", fontsize=12)
    
    ytick_fontsize = 10 if num_pairs <= 30 else 8
    ax2.tick_params(axis='y', labelsize=ytick_fontsize)

    ax3 = ax2.twinx()
    ax3.set_ylim(ax2.get_ylim())
    ax3.set_yticks(ax2.get_yticks())
    ax3.set_yticklabels([f"{occ:.2f}%" for occ in valid_occupancies])
    ax3.tick_params(axis='y', labelsize=ytick_fontsize)

    plt.tight_layout()
    
    plt.savefig(os.path.join(outdir, "pipi_all_panel.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, "pipi_all_panel.tif"), dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(os.path.join(outdir, "pipi_all_panel.svg"), format='svg', bbox_inches='tight')
    
    logging.info(f"[+] Analysis workflow completed successfully. High-res images (PNG/TIF/SVG) and data archived in './{outdir}/'.")

if __name__ == "__main__":
    main()
