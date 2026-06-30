#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import logging
import argparse
import itertools
import yaml
import textwrap
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import matplotlib.colors as mcolors
import concurrent.futures
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

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def setup_logger(output_dir):
    log_file = os.path.join(output_dir, "analysis_process.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w', encoding='utf-8')])
    return log_file

def parse_yaml_config(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def generate_target_pairs(pair_config):
    mode = pair_config.get('mode', 'explicit')
    def parse_res(res_input):
        if isinstance(res_input, int): return [str(res_input)]
        if isinstance(res_input, str): res_input = res_input.split(',')
        if not res_input: return []
        res = []
        for item in res_input:
            if isinstance(item, int): res.append(str(item))
            elif isinstance(item, str):
                item = item.strip()
                chain_prefix = ""
                if ':' in item:
                    chain_prefix, item = item.split(':', 1)
                    chain_prefix = chain_prefix + "_"
                
                if '-' in item:
                    start_str, end_str = item.split('-')
                    start, end = int(start_str), int(end_str)
                    step = 1 if start <= end else -1
                    res.extend([f"{chain_prefix}{i}" for i in range(start, end + step, step)])
                else:
                    res.append(f"{chain_prefix}{item}")
        return res

    if mode == 'exhaustive':
        list_A = parse_res(pair_config.get('list_A', []))
        list_B = parse_res(pair_config.get('list_B', []))
        return list(itertools.product(list_A, list_B))
    elif mode == 'explicit':
        if 'explicit_list' in pair_config: return [tuple(p) for p in pair_config['explicit_list']]
        else:
            list_A = parse_res(pair_config.get('list_A', []))
            list_B = parse_res(pair_config.get('list_B', []))
            return list(zip(list_A, list_B))
    else: raise ValueError(f"Unsupported mode: {mode}")

def build_pdb_and_dict(gmx_cmd, tpr, outdir):
    ref_pdb = os.path.join(outdir, "system_ref.pdb")
    if not os.path.exists(ref_pdb):
        subprocess.run([gmx_cmd, "editconf", "-f", tpr, "-o", ref_pdb], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    atom_dict = {}
    ignore_resnames = {"SOL", "WAT", "HOH", "NA", "CL", "K", "MG", "CA", "ZN", "ION", "TIP3P"}
    
    if os.path.exists(ref_pdb):
        with open(ref_pdb, "r") as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        serial = int(line[6:11].strip())
                        atomname = line[12:16].strip()
                        resname = line[17:21].strip()
                        chain = line[21].strip() if len(line) > 21 else ""
                        resnr = int(line[22:26].strip())
                        
                        if resname.upper() not in ignore_resnames:
                            res_key = f"{chain}_{resnr}" if chain else str(resnr)
                            atom_dict[serial] = {
                                "chain": chain, "resname": resname, "resnr": resnr, 
                                "atomname": atomname, "res_key": res_key
                            }
                    except: pass
    return atom_dict

def inspect_topology(gmx_cmd, tpr, traj, outdir):
    os.makedirs(outdir, exist_ok=True)
    print(f"\n{Colors.CYAN}{'='*70}\n GROMACS Topology Inspector (Real Topology Mapping)\n{'='*70}{Colors.RESET}")
    ref_pdb = os.path.join(outdir, "system_ref.pdb")
    if not os.path.exists(ref_pdb):
        print(f"{Colors.YELLOW}[*] Extracting structure from TPR... Please wait.{Colors.RESET}")
        subprocess.run([gmx_cmd, "editconf", "-f", tpr, "-o", ref_pdb], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    chains = {}
    with open(ref_pdb, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                serial = int(line[6:11].strip())
                chain = line[21].strip() or "[None]"
                resname = line[17:21].strip()
                resnr = int(line[22:26].strip())
                
                if chain not in chains: 
                    chains[chain] = {'res': [], 'min_atom': serial, 'max_atom': serial}
                
                chains[chain]['min_atom'] = min(chains[chain]['min_atom'], serial)
                chains[chain]['max_atom'] = max(chains[chain]['max_atom'], serial)
                
                if not chains[chain]['res'] or chains[chain]['res'][-1][1] != resnr:
                    chains[chain]['res'].append((resname, resnr))
                    
    for c, data in chains.items():
        res_list = data['res']
        first_res, first_id = res_list[0]
        last_res, last_id = res_list[-1]
        print(f" {Colors.GREEN}Chain {c:<6}{Colors.RESET} | {len(res_list):>4} Residues | Range: {first_res:<4} {first_id:>4}  -->  {last_res:<4} {last_id:>4} | Atoms: {data['min_atom']} - {data['max_atom']}")
    
    print(f"\n{Colors.YELLOW}[ Configuration Guide (Copy & Paste) ]: {Colors.RESET}")
    print(f"{Colors.YELLOW}The system has automatically generated the optimal config.yaml core configuration. Please copy and replace the content below!{Colors.RESET}\n")
    
    print(f"{Colors.RED}# =========================================={Colors.RESET}")
    print(f"{Colors.RED}# 0. Confirm current folder system paths (Easily missed!):{Colors.RESET}")
    print(f"{Colors.RED}# =========================================={Colors.RESET}")
    print(f"system:")
    print(f"  traj: \"{traj}\"   {Colors.YELLOW}# <-- Please confirm: Is this the actual xtc file name in the current directory?{Colors.RESET}")
    print(f"  tpr: \"{tpr}\"     {Colors.YELLOW}# <-- Please confirm: Is this the actual tpr file name in the current directory?{Colors.RESET}\n")

    valid_chains = [c for c in chains.keys() if c != "[None]"]
    
    if len(valid_chains) >= 2:
        if len(valid_chains) >= 3:
            c1, c2, c3 = valid_chains[0], valid_chains[1], valid_chains[2]
            c1_r_start, c1_r_end = chains[c1]['res'][0][1], chains[c1]['res'][-1][1]
            c2_r_start, c2_r_end = chains[c2]['res'][0][1], chains[c2]['res'][-1][1]
            c3_r_start, c3_r_end = chains[c3]['res'][0][1], chains[c3]['res'][-1][1]
            
            # V21 Core: Replaces Chain A syntax with absolute atomnr addressing
            g1_str = f"\"atomnr {chains[c1]['min_atom']} to {chains[c1]['max_atom']} or atomnr {chains[c2]['min_atom']} to {chains[c2]['max_atom']}\""
            g2_str = f"\"atomnr {chains[c3]['min_atom']} to {chains[c3]['max_atom']}\""
            
            p_a_str = f"\"{c1}:{c1_r_start}-{c1_r_end}, {c2}:{c2_r_start}-{c2_r_end}\""
            p_b_str = f"\"{c3}:{c3_r_start}-{c3_r_end}\""
            
            e1_r1, e1_r2 = chains[c1]['res'][-1][1], chains[c3]['res'][0][1]
            e2_r1, e2_r2 = chains[c2]['res'][-1][1], chains[c3]['res'][min(1, len(chains[c3]['res'])-1)][1]
            e3_r1, e3_r2 = chains[c1]['res'][max(0, len(chains[c1]['res'])-2)][1], chains[c3]['res'][min(2, len(chains[c3]['res'])-1)][1]
            
            exp_1 = f"[\"{c1}:{e1_r1}\", \"{c3}:{e1_r2}\"]    # Example: Chain {c1} and Chain {c3} pairing"
            exp_2 = f"[\"{c2}:{e2_r1}\", \"{c3}:{e2_r2}\"]    # Example: Chain {c2} and Chain {c3} pairing"
            exp_3 = f"[\"{c1}:{e3_r1}\", \"{c3}:{e3_r2}\"]"
            
            multi_note = f"{Colors.YELLOW}# ⚠️ Detected {len(valid_chains)} main chains, the system may have overlapping numbering!{Colors.RESET}\n" + \
                         f"{Colors.YELLOW}# Absolute 'atomnr' selection syntax has been generated for you in macro groups.{Colors.RESET}\n" + \
                         f"{Colors.YELLOW}# Using (Chain {c1} + Chain {c2}) against (Chain {c3}) as an example:{Colors.RESET}"
            
        else:
            c1, c2 = valid_chains[0], valid_chains[1]
            c1_r_start, c1_r_end = chains[c1]['res'][0][1], chains[c1]['res'][-1][1]
            c2_r_start, c2_r_end = chains[c2]['res'][0][1], chains[c2]['res'][-1][1]
            
            g1_str = f"\"atomnr {chains[c1]['min_atom']} to {chains[c1]['max_atom']}\""
            g2_str = f"\"atomnr {chains[c2]['min_atom']} to {chains[c2]['max_atom']}\""
            
            p_a_str = f"\"{c1}:{c1_r_start}-{c1_r_end}\""
            p_b_str = f"\"{c2}:{c2_r_start}-{c2_r_end}\""
            
            e1_r1, e1_r2 = chains[c1]['res'][min(0, len(chains[c1]['res'])-1)][1], chains[c2]['res'][min(0, len(chains[c2]['res'])-1)][1]
            e2_r1, e2_r2 = chains[c1]['res'][min(1, len(chains[c1]['res'])-1)][1], chains[c2]['res'][min(1, len(chains[c2]['res'])-1)][1]
            e3_r1, e3_r2 = chains[c1]['res'][min(2, len(chains[c1]['res'])-1)][1], chains[c2]['res'][min(2, len(chains[c2]['res'])-1)][1]
            
            exp_1 = f"[\"{c1}:{e1_r1}\", \"{c2}:{e1_r2}\"]    # Example: Forced pairing of Chain {c1} and Chain {c2}"
            exp_2 = f"[\"{c1}:{e2_r1}\", \"{c2}:{e2_r2}\"]    # And so on..."
            exp_3 = f"[\"{c1}:{e3_r1}\", \"{c2}:{e3_r2}\"]"
            
            multi_note = f"{Colors.YELLOW}# 💡 Macro groups use 'atomnr' absolute coordinates, preventing GROMACS chain identification bugs.{Colors.RESET}"

        print(f"{Colors.GREEN}# =========================================={Colors.RESET}")
        print(f"{Colors.GREEN}# 1. Replace the 'groups' section in yaml:{Colors.RESET}")
        print(f"{Colors.GREEN}# =========================================={Colors.RESET}")
        if multi_note: print(multi_note)
        print(f"groups:")
        print(f"  group1: {g1_str}")
        print(f"  group2: {g2_str}\n")
        
        print(f"{Colors.GREEN}# =========================================={Colors.RESET}")
        print(f"{Colors.GREEN}# 2. Replace the 'pairs' section in yaml (Exhaustive large-scale mode):{Colors.RESET}")
        print(f"{Colors.GREEN}# =========================================={Colors.RESET}")
        if len(valid_chains) >= 3: print(f"{Colors.YELLOW}# 💡 Tip: You can merge multiple chains into the same group by separating them with commas in list_A!{Colors.RESET}")
        print(f"pairs:")
        print(f"  mode: \"exhaustive\"")
        print(f"  list_A: {p_a_str}")
        print(f"  list_B: {p_b_str}\n")
        
        print(f"{Colors.YELLOW}# =========================================={Colors.RESET}")
        print(f"{Colors.YELLOW}# 💡 Advanced: If you wish to use Explicit (precise one-to-one) mode{Colors.RESET}")
        print(f"{Colors.YELLOW}# =========================================={Colors.RESET}")
        print(f"{Colors.YELLOW}# Suitable for known definitive targets (e.g., ligand binding core sites, specific base pairs).{Colors.RESET}")
        print(f"{Colors.YELLOW}# Please avoid the exhaustive configuration above and use the following format:{Colors.RESET}")
        print(f"{Colors.YELLOW}pairs:{Colors.RESET}")
        print(f"{Colors.YELLOW}  mode: \"explicit\"{Colors.RESET}")
        print(f"{Colors.YELLOW}  explicit_list:{Colors.RESET}")
        print(f"{Colors.YELLOW}    - {exp_1}{Colors.RESET}")
        print(f"{Colors.YELLOW}    - {exp_2}{Colors.RESET}")
        print(f"{Colors.YELLOW}    - {exp_3}{Colors.RESET}\n")
        
    elif len(valid_chains) == 1:
        c1 = valid_chains[0]
        r1_start, r1_end = chains[c1]['res'][0][1], chains[c1]['res'][-1][1]
        print(f"{Colors.YELLOW}# Tip: Only one main chain detected. If docking with a ligand, verify if the ligand is classified in the [None] unassigned chain group.{Colors.RESET}")
        print(f"list_A: \"{c1}:{r1_start}-{r1_end}\"\n")
    else:
        print(f"{Colors.RED}Your topology core molecule does not contain chain ID information. Please manually select numbering as needed.{Colors.RESET}\n")

    sys.exit(0)

def perform_fast_discovery(gmx_cmd, traj, tpr, base_pairs, atom_dict, outdir, force):
    err_msg = ""
    res_A = set([p[0] for p in base_pairs])
    res_B = set([p[1] for p in base_pairs])
    
    atoms_A = [s for s, info in atom_dict.items() if info['res_key'] in res_A or str(info['resnr']) in res_A]
    atoms_B = [s for s, info in atom_dict.items() if info['res_key'] in res_B or str(info['resnr']) in res_B]
    
    if not atoms_A or not atoms_B:
        return [], "No atoms found in dictionary for the specified residues/chains."
        
    ndx_name = os.path.join(outdir, "fast_discovery_macro.ndx")
    with open(ndx_name, "w") as f:
        f.write("[ Side_A ]\n")
        for i in range(0, len(atoms_A), 15): f.write(" ".join(map(str, atoms_A[i:i+15])) + "\n")
        f.write("[ Side_B ]\n")
        for i in range(0, len(atoms_B), 15): f.write(" ".join(map(str, atoms_B[i:i+15])) + "\n")

    out_ndx = os.path.join(outdir, "fast_discovered_pairs.ndx")
    out_xvg = os.path.join(outdir, "fast_discovered_num.xvg")
    
    if not os.path.exists(out_ndx):
        cmd = [gmx_cmd, "hbond", "-f", traj, "-s", tpr, "-n", ndx_name, 
               "-r", '"Side_A"', "-t", '"Side_B"', 
               "-num", out_xvg, "-o", out_ndx]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip()
            if not err_msg: err_msg = "GROMACS Discovery Calculation Failed."
            return [], err_msg

    pairs = set()
    if os.path.exists(out_ndx) and os.path.getsize(out_ndx) > 0:
        try:
            with open(out_ndx, 'r') as f:
                content = f.read()
            for sec in content.split('['):
                if 'hbonds' in sec.lower() or 'pairs' in sec.lower():
                    lines = sec.split('\n')[1:]
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 3:
                            pairs.add((int(parts[0]), int(parts[1]), int(parts[2])))
                        elif len(parts) == 2:
                            pairs.add((int(parts[0]), None, int(parts[1])))
        except Exception as e:
            err_msg = f"Parsing Error: {str(e)}"
    
    return list(pairs), err_msg

def run_pair_hbond(task):
    gmx_cmd, traj, tpr, sel1, sel2, name, out_xvg, out_ndx, out_dist, out_ang, force = task
    
    if not os.path.exists(out_xvg):
        cmd = [gmx_cmd, "hbond", "-f", traj, "-s", tpr, "-r", sel1, "-t", sel2, 
               "-num", out_xvg, "-o", out_ndx, "-dist", out_dist, "-ang", out_ang]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            return True, name, ""
        except subprocess.CalledProcessError as e:
            err_details = [line for line in e.stderr.split('\n') if "Error" in line or "error" in line or "Exception" in line]
            err_str = err_details[-1] if err_details else e.stderr.split('\n')[-2]
            return False, name, err_str
    return True, name, ""

def parse_xvg(xvg_file):
    times, hbonds = [], []
    with open(xvg_file, 'r') as f:
        for line in f:
            if not line.startswith(('@', '#')) and len(line.strip().split()) == 2:
                parts = line.strip().split()
                times.append(float(parts[0]) / 1000.0) 
                hbonds.append(int(parts[1]))
    return np.array(times), np.array(hbonds)

def parse_hist_xvg(xvg_file):
    if not os.path.exists(xvg_file): return np.array([]), np.array([]), 0.0
    x, y = [], []
    with open(xvg_file, 'r') as f:
        for line in f:
            if not line.startswith(('@', '#')) and len(line.strip().split()) >= 2:
                parts = line.strip().split()
                x.append(float(parts[0]))
                y.append(float(parts[1]))
    x, y = np.array(x), np.array(y)
    if np.sum(y) == 0: return x, y, 0.0
    avg = np.average(x, weights=y)
    return x, y, avg

def reconstruct_population(x, y, samples=2000):
    if np.sum(y) == 0: return []
    y_norm = y / np.sum(y)
    counts = np.round(y_norm * samples).astype(int)
    data = []
    for val, count in zip(x, counts):
        data.extend([val] * count)
    return data

def main():
    parser = argparse.ArgumentParser(description="GROMACS H-Bond Analysis Tool (V21 Ultimate Edition)")
    parser.add_argument("config", nargs="?", default="config.yaml", help="Path to config file")
    parser.add_argument("-a", "--atoms", action="store_true", help="Perform atomic-level analysis")
    parser.add_argument("-i", "--inspect", action="store_true", help="Launch Topology Inspector mode to map real residues")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"\n{Colors.YELLOW}Error: Config file not found!{Colors.RESET}\n")
        sys.exit(1)

    cfg = parse_yaml_config(args.config)
    sys_cfg, plot_cfg = cfg['system'], cfg.get('plot', {})
    
    gmx_cmd = sys_cfg.get('gmx_cmd', 'gmx')
    traj = sys_cfg.get('traj', 'fit.xtc')
    tpr = sys_cfg.get('tpr', 'md_0_1.tpr')
    outdir = sys_cfg.get('outdir', 'hbond_out')
    force = sys_cfg.get('force', False)
    
    missing_files = []
    if not os.path.exists(traj): missing_files.append(f"Trajectory file (traj) -> {traj}")
    if not os.path.exists(tpr): missing_files.append(f"Topology file (tpr) -> {tpr}")
        
    if missing_files:
        print(f"\n{Colors.RED}{'='*70}")
        print(f"[!] Fatal Error: Core files specified in config.yaml were not found in the current directory!")
        for mf in missing_files:
            print(f"    ❌ Missing: {mf}")
        print(f"\n💡 Solution: Open config.yaml and modify 'traj' and 'tpr' to match the actual file names in the current directory.")
        print(f"{'='*70}{Colors.RESET}\n")
        sys.exit(1)

    safe_outdir = os.path.abspath(outdir)
    current_dir = os.path.abspath(".")
    if safe_outdir == current_dir or safe_outdir == os.path.abspath("/") or not outdir:
        print(f"\n{Colors.RED}[!] Critical Security Warning: 'outdir' must absolutely not be set to the current directory (.) or root directory (/)! Execution forcefully aborted to prevent accidental deletion.{Colors.RESET}\n")
        sys.exit(1)

    if force and os.path.exists(outdir):
        print(f"{Colors.YELLOW}[*] Force Mode [ON]: Thoroughly wiping output directory '{outdir}' for a fresh start...{Colors.RESET}")
        shutil.rmtree(outdir)

    os.makedirs(outdir, exist_ok=True)

    if args.inspect:
        ref_pdb = os.path.join(outdir, "system_ref.pdb")
        if os.path.exists(ref_pdb):
            try: os.remove(ref_pdb)
            except OSError: pass
        inspect_topology(gmx_cmd, tpr, traj, outdir)
    
    group1 = cfg['groups'].get('group1', 'Protein')
    group2 = cfg['groups'].get('group2', 'DNA')
    
    occ_threshold = float(plot_cfg.get('occ_threshold', 10.0))
    kde_threshold = float(plot_cfg.get('kde_threshold', 75.0)) 
    clean_threshold = float(plot_cfg.get('clean_threshold', occ_threshold))
    cbar_max = plot_cfg.get('cbar_max', 3)
    c_min_rgb, c_max_rgb = plot_cfg.get('color_min', [255, 255, 255]), plot_cfg.get('color_max', [63, 169, 245])

    is_atomic = args.atoms
    label_prefix = "Atom" if is_atomic else "Pair"
    analysis_mode = "ATOMIC-Level" if is_atomic else "RESIDUE-Level"

    base_pairs = generate_target_pairs(cfg['pairs'])
    log_file = setup_logger(outdir)
    
    print(f"\n{Colors.CYAN}{'='*60}\n Starting Config-Driven H-Bond Analysis ({analysis_mode})\n{'='*60}{Colors.RESET}\n")
    print(f"[*] Config Loaded : {args.config}")
    logging.info(f"=== Starting Config-Driven H-Bond Analysis ({analysis_mode}) ===")

    print(f"{Colors.YELLOW}[*] Initializing PDB Map and Atom Dictionary for Smart Screening...{Colors.RESET}")
    atom_dict = build_pdb_and_dict(gmx_cmd, tpr, outdir)
    if not atom_dict:
        print(f"{Colors.RED}[!] Failed to generate system_ref.pdb. Aborting fast discovery.{Colors.RESET}")
        sys.exit(1)

    final_targets = []
    is_explicit = (cfg['pairs'].get('mode', 'explicit') == 'explicit')
    
    if is_explicit and not is_atomic:
        print(f"{Colors.YELLOW}[*] Mode [Explicit] Detected: Bypassing Fast Discovery to guarantee reporting of 0% pairs.{Colors.RESET}")
        for r1, r2 in base_pairs:
            name1 = r1.replace("_", "") if "_" in r1 else r1
            name2 = r2.replace("_", "") if "_" in r2 else r2
            orig_r1 = r1.split('_')[1] if "_" in r1 else r1
            orig_r2 = r2.split('_')[1] if "_" in r2 else r2
            
            final_targets.append({
                'name': f"{name1}-{name2}", 
                'sel1': f"resid {orig_r1}", 
                'sel2': f"resid {orig_r2}"
            })
        print(f"{Colors.GREEN} [+] Explicit Residue Tasks Prepared: {len(final_targets)} pairs.{Colors.RESET}")

    else:
        print(f"{Colors.YELLOW}[*] Phase 1: One-Shot Trajectory Pre-screening (Fast Discovery)...{Colors.RESET}")
        raw_discovered_atoms, phase1_err = perform_fast_discovery(gmx_cmd, traj, tpr, base_pairs, atom_dict, outdir, force)
        if phase1_err and not raw_discovered_atoms:
            print(f"\n{Colors.RED}[!] Phase 1 Discovery Failed:\n{phase1_err}{Colors.RESET}")
            sys.exit(1)

        if not is_atomic:
            discovered_res_pairs = set()
            for d, h, a in raw_discovered_atoms:
                if d in atom_dict and a in atom_dict:
                    r_d = atom_dict[d]['res_key']
                    r_a = atom_dict[a]['res_key']
                    discovered_res_pairs.add((r_d, r_a))
                    discovered_res_pairs.add((r_a, r_d))
                    
            valid_res_pairs = []
            for p1, p2 in base_pairs:
                match_p1 = [p1] if "_" in p1 else [p1, f"A_{p1}", f"B_{p1}", f"C_{p1}", f"D_{p1}", f"X_{p1}"]
                match_p2 = [p2] if "_" in p2 else [p2, f"A_{p2}", f"B_{p2}", f"C_{p2}", f"D_{p2}", f"X_{p2}"]
                
                found = False
                for mp1 in match_p1:
                    for mp2 in match_p2:
                        if (mp1, mp2) in discovered_res_pairs:
                            valid_res_pairs.append((mp1, mp2))
                            found = True
                            break
                    if found: break
                    
            for r1, r2 in valid_res_pairs:
                name1 = r1.replace("_", "") if "_" in r1 else r1
                name2 = r2.replace("_", "") if "_" in r2 else r2
                orig_r1 = r1.split('_')[1] if "_" in r1 else r1
                orig_r2 = r2.split('_')[1] if "_" in r2 else r2
                
                final_targets.append({
                    'name': f"{name1}-{name2}", 
                    'sel1': f"resid {orig_r1}", 
                    'sel2': f"resid {orig_r2}"
                })
                
            print(f"{Colors.GREEN} [+] Phase 1 Complete! Reduced to {len(final_targets)} real RESIDUE tasks.{Colors.RESET}")

        else:
            base_pairs_set = set(base_pairs)
            for d, h, a in raw_discovered_atoms:
                if d in atom_dict and a in atom_dict:
                    iD, iA = atom_dict[d], atom_dict[a]
                    keys_d = {iD['res_key'], str(iD['resnr'])}
                    keys_a = {iA['res_key'], str(iA['resnr'])}
                    
                    match = False
                    for kd in keys_d:
                        for ka in keys_a:
                            if (kd, ka) in base_pairs_set or (ka, kd) in base_pairs_set:
                                match = True
                                break
                        if match: break
                                
                    if match:
                        prefix_d = f"{iD['chain']}:" if iD['chain'] else ""
                        prefix_a = f"{iA['chain']}:" if iA['chain'] else ""
                        name_str = f"{prefix_d}{iD['resname']}{iD['resnr']}({iD['atomname']}) - {prefix_a}{iA['resname']}{iA['resnr']}({iA['atomname']})"
                        
                        sel1 = f"atomnr {d} {h}" if h else f"atomnr {d}"
                        sel2 = f"atomnr {a}"
                        final_targets.append({'name': name_str, 'sel1': sel1, 'sel2': sel2})
                        
            unique_targets, seen = [], set()
            for t in final_targets:
                if t['name'] not in seen:
                    seen.add(t['name'])
                    unique_targets.append(t)
            final_targets = unique_targets
            print(f"{Colors.GREEN} [+] Phase 1 Complete! Fissioned into {len(final_targets)} precise ATOMIC tasks.{Colors.RESET}")

    if not final_targets:
        print(f"\n{Colors.RED}[!] Error: No targets generated! No hydrogen bonds formed in these regions.{Colors.RESET}")
        sys.exit(0)

    final_targets.sort(key=lambda x: natural_sort_key(x['name']))

    print(f"\n{Colors.YELLOW}[*] Calculating macro groups ({group1} vs {group2})...{Colors.RESET}")
    total_xvg, total_ndx = os.path.join(outdir, "hbnum_total.xvg"), os.path.join(outdir, "hbond_total.ndx")
        
    if not os.path.exists(total_xvg):
        cmd = [gmx_cmd, "hbond", "-f", traj, "-s", tpr, "-r", group1, "-t", group2, "-num", total_xvg, "-o", total_ndx]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip()
            print(f"\n{Colors.RED}[!] Macro Group Calculation Failed{Colors.RESET}")
            print(f"\n{Colors.YELLOW}💡 Diagnosis:{Colors.RESET}")
            print(f"1. [Trajectory & Topology Mismatch]: Your {traj} might be stripped of water, but {tpr} contains many water molecules!")
            print(f"   When using global dynamic variables like 'chain', GROMACS strictly scans the entire topology, leading to out-of-bounds addressing for non-existent atoms.")
            print(f"   👉 Solution: Run `gmx convert-tpr -s {tpr} -o match.tpr` in the terminal to extract a clean tpr and update the yaml.")
            print(f"2. [Selection Syntax Error]: GROMACS may not recognize '{group1}' or '{group2}'.")
            print(f"   👉 Solution: Check spelling, or run with `--inspect` to obtain absolute 'atomnr' coordinate syntax and replace it in the yaml.")
            print(f"\n{Colors.RED}👇 Detailed GROMACS Native Error Log 👇{Colors.RESET}")
            print(f"{Colors.RED}{err_msg}{Colors.RESET}")
            sys.exit(1)

    times_total, hbonds_total = parse_xvg(total_xvg)
    total_frames = len(times_total)

    tasks = []
    for t in final_targets:
        name = t['name']
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace(":", "")
        xvg_name, ndx_name = os.path.join(outdir, f"hbnum_{safe_name}.xvg"), os.path.join(outdir, f"hbond_{safe_name}.ndx")
        dist_name, ang_name = os.path.join(outdir, f"hbdist_{safe_name}.xvg"), os.path.join(outdir, f"hbang_{safe_name}.xvg")
        tasks.append((gmx_cmd, traj, tpr, t['sel1'], t['sel2'], name, xvg_name, ndx_name, dist_name, ang_name, force))

    prefix_str = "Phase 2" if is_atomic else "Calculation"
    print(f"{Colors.YELLOW}[*] [{prefix_str}] Dispatching tasks to GROMACS...{Colors.RESET}")
    successful_targets, failed_targets_with_err = [], []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(run_pair_hbond, tasks), total=len(tasks), desc="Processing", ascii=False, ncols=80))
        for success, name, err in results:
            if success: successful_targets.append(name)
            else: failed_targets_with_err.append((name, err))

    if failed_targets_with_err: 
        print(f"\n{Colors.RED}[!] Auto-Skipped {len(failed_targets_with_err)} targets (See log for details).{Colors.RESET}")

    if not successful_targets: sys.exit(1)
    
    successful_targets.sort(key=natural_sort_key)
    print(f"\n{Colors.GREEN}[+] All valid GROMACS calculations completed.{Colors.RESET}\n")

    all_pair_data, valid_occupancies = {}, []
    valid_avg_h, valid_avg_d, valid_avg_a, valid_pair_names = [], [], [], []
    violin_data, elite_dist_ang_data = [], {}

    print(f"{Colors.CYAN}{'='*80}\n {'Hydrogen Bond Comprehensive Report':^78}\n{'='*80}{Colors.RESET}")
    logging.info("\n=== Hydrogen Bond Comprehensive Report ===")

    for name in successful_targets:
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace(":", "")
        xvg_name = os.path.join(outdir, f"hbnum_{safe_name}.xvg")
        dist_name = os.path.join(outdir, f"hbdist_{safe_name}.xvg")
        ang_name = os.path.join(outdir, f"hbang_{safe_name}.xvg")
        
        _, hbonds = parse_xvg(xvg_name)
        dist_x, dist_y, avg_d = parse_hist_xvg(dist_name)
        ang_x, ang_y, avg_a = parse_hist_xvg(ang_name)
        
        occ = (np.sum(hbonds >= 1) / total_frames) * 100
        avg_h = np.mean(hbonds)
        all_pair_data[name] = {'occ': occ, 'hbonds': hbonds}
        
        msg = (f"Target {name:<30}: Occ = {occ:>.3g}% | Avg H-Bonds = {avg_h:.3g} | Dist = {avg_d:.3g} nm | Ang = {avg_a:>.3g}°")
        print(msg)
        logging.info(msg)

        if occ >= occ_threshold:
            valid_pair_names.append(name)
            valid_occupancies.append(occ)
            valid_avg_h.append(avg_h)
            valid_avg_d.append(avg_d)
            valid_avg_a.append(avg_a)
            pop = reconstruct_population(dist_x, dist_y)
            for d_val in pop:
                violin_data.append({"Target": name, "Distance": d_val})
                
        if occ >= kde_threshold:
            elite_dist_ang_data[name] = {'dx': dist_x, 'dy': dist_y, 'ax': ang_x, 'ay': ang_y, 'occ': occ}

    df_violin = pd.DataFrame(violin_data)
    
    print(f"\n{Colors.YELLOW}[*] Auto-Cleaning: Removing temporary files for pairs (< {clean_threshold}% Occ)...{Colors.RESET}")
    cleaned_count = 0
    for name in successful_targets:
        if all_pair_data[name]['occ'] < clean_threshold:
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace(":", "")
            for suffix in ["hbnum_", "hbdist_", "hbang_", "hbond_"]:
                ext = ".ndx" if suffix == "hbond_" else ".xvg"
                fpath = os.path.join(outdir, f"{suffix}{safe_name}{ext}")
                try:
                    if os.path.exists(fpath): os.remove(fpath)
                except OSError: pass
            cleaned_count += 1
    if cleaned_count > 0:
        print(f"    -> Successfully purged files for {cleaned_count} irrelevant targets to save disk space.")
    else:
        print(f"    -> No files needed cleaning based on the {clean_threshold}% threshold.")

    if not valid_pair_names:
        print(f"\n{Colors.RED}[!] Error: No data reached {occ_threshold}%!{Colors.RESET}")
        sys.exit(1)

    print(f"\n{Colors.CYAN}[*] Filtering applied: {len(valid_pair_names)} items kept for plotting (Occ >= {occ_threshold}%).{Colors.RESET}")
    print(f"{Colors.YELLOW}[*] Exporting raw plotting data to CSV...{Colors.RESET}")
    
    barcode_matrix = np.array([all_pair_data[n]['hbonds'] for n in valid_pair_names])
    pd.DataFrame({'Time_ns': times_total, 'Total_Hbonds': hbonds_total}).to_csv(os.path.join(outdir, "plotdata_1_total.csv"), index=False, float_format='%.3f')
    df_plot2 = pd.DataFrame(barcode_matrix.T, columns=valid_pair_names)
    df_plot2.insert(0, 'Time_ns', times_total)
    df_plot2.to_csv(os.path.join(outdir, "plotdata_2_barcode.csv"), index=False, float_format='%.3f')
    if not df_violin.empty: df_violin.to_csv(os.path.join(outdir, "plotdata_3_violin.csv"), index=False, float_format='%.3f')
    pd.DataFrame({'Target': valid_pair_names, 'Occ_%': valid_occupancies, 'Dist_nm': valid_avg_d, 'Hbonds': valid_avg_h}).to_csv(os.path.join(outdir, "plotdata_4_bubble.csv"), index=False, float_format='%.3f')

    print(f"{Colors.YELLOW}[*] Generating visualization dashboard...{Colors.RESET}")
    title_suffix = " (Atomic Level Tracking)" if is_atomic else " (Residue Level Tracking)"
    ylabel_name = "Target Atom Pairs" if is_atomic else "Target Residue Pairs"

    num_pairs = len(valid_pair_names)
    dynamic_height = min(40, max(18, 10 + num_pairs * 0.4))
    fig = plt.figure(figsize=(14, dynamic_height))
    c_min, c_max = [x / 255.0 for x in c_min_rgb], [x / 255.0 for x in c_max_rgb]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [c_min, c_max])

    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(times_total, hbonds_total, color=tuple(c_max), linewidth=1.5, alpha=0.8)
    global_avg = np.mean(hbonds_total)
    ax1.axhline(y=global_avg, color='gray', linestyle='--', linewidth=1.5)
    ax1.text(times_total[-1]*0.99, global_avg + 0.1, f' Avg: {global_avg:.3g} ', color='black', va='bottom', ha='right', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', pad=2))
    ax1.set_title(f"1. Total Hydrogen Bonds over Time{title_suffix}", fontsize=16, pad=10, weight='bold')
    ax1.set_ylabel("Total H-bonds", fontsize=12)
    ax1.set_xlim(times_total[0], times_total[-1])
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = plt.subplot(4, 1, 2)
    sns.heatmap(barcode_matrix, cmap=custom_cmap, vmin=0, vmax=cbar_max, cbar_kws={'label': 'Number of H-bonds'}, yticklabels=valid_pair_names, ax=ax2)
    num_ticks = 6
    tick_positions = np.linspace(0, barcode_matrix.shape[1] - 1, num_ticks)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([f"{times_total[int(idx)]:.3f}" for idx in tick_positions], rotation=0)
    ax2.set_title(f"2. Hydrogen Bond Dynamics (Filtered > {occ_threshold}%){title_suffix}", fontsize=16, pad=10, weight='bold')
    ax2.set_xlabel("Time (ns)", fontsize=12)
    ax2.set_ylabel(ylabel_name, fontsize=12)
    ytick_fontsize = 10 if num_pairs <= 30 else 8
    ax2.tick_params(axis='y', labelsize=ytick_fontsize)

    ax3 = ax2.twinx()
    ax3.set_ylim(ax2.get_ylim())
    ax3.set_yticks(ax2.get_yticks())
    ax3.set_yticklabels([f"{occ:.3g}%" for occ in valid_occupancies])
    ax3.tick_params(axis='y', labelsize=ytick_fontsize)

    ax3_plot = plt.subplot(4, 1, 3)
    if not df_violin.empty:
        sns.violinplot(x="Target", y="Distance", data=df_violin, color=tuple(c_max), inner="quartile", ax=ax3_plot, linewidth=1)
    cutoff_line = ax3_plot.axhline(y=0.35, color='red', linestyle=':', label='0.35 nm (GMX Cutoff)')
    quartile_legend = mlines.Line2D([], [], color='k', linestyle='--', label='Quartiles (25%, Median, 75%)')
    ax3_plot.set_title(f"3. Geometric Strength Distribution (Distance){title_suffix}", fontsize=16, pad=10, weight='bold')
    ax3_plot.set_ylabel("Distance (nm)", fontsize=12)
    ax3_plot.set_xlabel(ylabel_name, fontsize=12)
    ax3_plot.legend(handles=[cutoff_line, quartile_legend], loc='upper right')
    ax3_plot.tick_params(axis='x', rotation=45, labelsize=9)

    ax4 = plt.subplot(4, 1, 4)
    x_val, y_val = valid_occupancies, valid_avg_d
    sizes = [h * 300 for h in valid_avg_h] 
    ax4.scatter(x_val, y_val, s=sizes, color=tuple(c_max), alpha=0.6, edgecolors='black')
    for i, txt in enumerate(valid_pair_names):
        ax4.text(x_val[i], y_val[i] + 0.002, txt, fontsize=9, ha='center', va='bottom', weight='bold', color='darkblue')
    ax4.set_title(f"4. Core Anchor Radar (Occupancy vs Distance){title_suffix}", fontsize=16, pad=10, weight='bold')
    ax4.set_xlabel("Occupancy (%) -> Higher is more stable", fontsize=12)
    ax4.set_ylabel("Average Distance (nm) -> Lower is stronger", fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.6)
    if y_val: ax4.set_ylim(max(y_val) + 0.01, min(y_val) - 0.01)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hbond_all.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, "hbond_all.tif"), dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig(os.path.join(outdir, "hbond_all.svg"), format='svg', bbox_inches='tight')

    if elite_dist_ang_data:
        print(f"{Colors.YELLOW}[*] Generating Elite Targets Geometric Matrix (>={kde_threshold}%)...{Colors.RESET}")
        N = len(elite_dist_ang_data)
        cols = 3 if N >= 3 else N
        rows = int(np.ceil(N / cols))
        fig2, axes2 = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        if N == 1: axes2 = np.array([axes2])
        axes2 = axes2.flatten()
        
        sorted_elite_keys = sorted(elite_dist_ang_data.keys(), key=natural_sort_key)
        
        for i, name in enumerate(sorted_elite_keys):
            data = elite_dist_ang_data[name]
            ax_dist = axes2[i]
            
            best_dist = data['dx'][np.argmax(data['dy'])]
            best_ang = data['ax'][np.argmax(data['ay'])]
            
            ax_dist.fill_between(data['dx'], data['dy'], color=tuple(c_max), alpha=0.5, label='Distance (nm)')
            ax_dist.plot(data['dx'], data['dy'], color=tuple(c_max), linewidth=2)
            ax_dist.set_xlabel("Distance (nm)", color='darkblue', fontweight='bold')
            ax_dist.set_ylabel("Probability (Distance)", color='darkblue')
            ax_dist.tick_params(axis='x', colors='darkblue')
            ax_dist.tick_params(axis='y', colors='darkblue')
            ax_dist.set_xlim(0.25, 0.36)
            ax_dist.axvline(best_dist, color='darkblue', linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax_ang = ax_dist.twiny().twinx()
            ax_ang.plot(data['ax'], data['ay'], color='red', linewidth=2, linestyle='-', label='Angle (°)')
            ax_ang.set_xlabel("Angle (Degree)", color='red', fontweight='bold')
            ax_ang.set_ylabel("Probability (Angle)", color='red')
            ax_ang.tick_params(axis='x', colors='red')
            ax_ang.tick_params(axis='y', colors='red')
            ax_ang.set_xlim(0, 45) 
            ax_ang.axvline(best_ang, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            
            ax_dist.set_title(f"{name}\n(Occ: {data['occ']:.3g}%)", fontsize=11, weight='bold', pad=15)
            
            box_text = f"Best Dist: {best_dist:.3g} nm\nBest Angle: {best_ang:.3g}°"
            ax_dist.text(0.95, 0.95, box_text, transform=ax_dist.transAxes,
                         fontsize=10, fontweight='bold', va='top', ha='right',
                         bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.5'))
            
        for j in range(N, len(axes2)):
            axes2[j].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "hbond_dist_bang.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(outdir, "hbond_dist_bang.tif"), dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
        plt.savefig(os.path.join(outdir, "hbond_dist_bang.svg"), format='svg', bbox_inches='tight')

    print(f"\n{Colors.GREEN}[+] All Done! Images and logs saved in './{outdir}/'.{Colors.RESET}\n")

if __name__ == "__main__":
    main()
