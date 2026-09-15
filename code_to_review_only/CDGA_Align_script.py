## ====================================================================================================
#                                     CDGA ALIGN SCRIPT 1.0.0
## ====================================================================================================
# CONTAINS CDGA ALIGN MODULE + HOST CHARACTERIZATION APPENDED MODULE 
# PART OF CDGA ALIGN 1.0.0 WORKFLOW IN KNIME ANALITICS PLATFORM
#
# Authors: Napiórkowska and Szeleszczuk
#
# Year: 2026
#
# If you use CDGA Tool in a published work, please cite the following paper accepted for publication:
# Napiórkowska E, Szeleszczuk Ł. CDGA: A Host-Based Alignment and Descriptor Analysis Tool for Cyclodextrin Inclusion Complexes.
# Journal of Chemical Information and Modeling, DOI: 10.1021/acs.jcim.6c02866.
#
# The Python scripts used in the CDGA workflows were developed with the assistance of ChatGPT: 
# (CDGA ALIGN MODULE: GPT-5.5 Thinking; OpenAI, HOST CHARACTERIZATION MODULE: GPT 5.6 Sol, OpenAI). 
# All AI-generated code was reviewed, tested, and verified by the authors for accuracy, reliability, 
# and reproducibility.
# The authors take full responsibility for the final implementation and its scientific validity.
#
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# CDGA uses third-party open-source libraries through their public APIs,
# including RDKit, NumPy, and pandas. These dependencies are distributed under their respective
# licenses.
#
# See the README.md file for details in the GitHub repository associated with this workflow.
# The GitHub repository also provides test datasets, example output files, and a user manual 
# explaining the output columns and diagnostic statuses.
# 
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
## =======================================
# CDGA ALIGN MODULE
## =======================================
#
# This module aligns cyclodextrin host guest complexes to a common
# reference structure and characterizes the position of the guest
# within the cyclodextrin cavity.
#
# Input:
#
#   RDKit molecules containing a cyclodextrin host and a guest
#   molecule with 3D coordinates.
#
# The first molecule in the input table is used as the reference.
#
# Host and guest identification:
#
# The host is expected to be a native alpha, beta, or gamma
# cyclodextrin.
#
# The input structure may also contain a guest molecule and additional
# disconnected species, such as water molecules or ions.
#
#   The largest molecular fragment is treated as the cyclodextrin host.
#   The largest remaining fragment is treated as the guest.
#
# Initial cyclodextrin alignment:
#
# The primary and secondary cyclodextrin rims are identified using
# SMARTS patterns that recognize characteristic CH2O and CHO
# connectivity in the cyclodextrin host.
#
#   A cavity coordinate frame is constructed using:
#
#       z axis   direction between the primary and secondary rims
#       x axis   principal direction obtained from PCA of the
#                projected primary rim atoms
#       y axis   direction perpendicular to x and z
#
#   The host centroid and cavity axes of each tested complex are
#   initially aligned to those of the reference.
#
#   Two orientations differing by 180 degrees around the cavity axis
#   are evaluated, and the orientation giving the better primary rim
#   overlap is selected.
#
# Host refinement:
#
#   Carbon and oxygen atoms belonging to the cyclodextrin ring core
#   are compared between the reference and tested host.
#
#   MCS matching is used to identify possible corresponding host atoms.
#
#   Symmetry related mappings are evaluated using Kabsch fitting, and
#   the correspondence giving the lowest host RMSD is selected.
#
#   The selected Kabsch transformation is applied to the complete
#   host guest complex.
#
# Guest characterization:
#
#   Guest depth is calculated relative to both the local cyclodextrin
#   midpoint and the reference cyclodextrin midpoint.
#
#   Positive depth corresponds to displacement toward the primary
#   O6 side of the cyclodextrin cavity.
#
#   Guest orientation relative to the cavity axis is calculated from
#   the principal molecular axis obtained by PCA.
#
# Guest comparison:
#
#   Guest atoms are matched to the reference using progressively more
#   relaxed structural matching if an exact match is not available.
#
#   Guest RMSD is calculated directly in the aligned host coordinate
#   system without applying an additional Kabsch fit to the guest.
#
#   Therefore, guest RMSD represents a change in binding pose relative
#   to the reference host rather than an independently optimized guest
#   superposition.
#
# Output:
#
#   CDGA_Align_molfiles/
#       aligned complex MOL files
#       Matching_atoms/
#
# The KNIME output table contains host alignment metrics, guest depth,
# guest orientation angle relative to the cyclodextrin cavity axis,
# guest RMSD, atom mapping information, and diagnostic status
# information for each processed complex.
# =======================================================================
#
import knime.scripting.io as knio
import pandas as pd
import numpy as np
import os
import re
import tempfile
import itertools
from rdkit import Chem
from rdkit.Chem import rdFMCS

df = knio.input_tables[0].to_pandas()
mol_col = "Molecules (RDKit Mol)"
mols = df[mol_col].tolist()

MIN_GUEST_ATOMS = 3
MIN_HOST_CORE_MCS_ATOMS = 12
GUEST_MATCH_TIMEOUT = 12
HOST_REFINE_TIMEOUT = 20


def safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", s)
    s = re.sub(r"\s+", "_", s).strip("._ ")
    return s if s else "unnamed"


def validate_mol_has_3d(m, label="molecule"):
    if m is None:
        raise ValueError(f"{label} is missing")
    if m.GetNumConformers() == 0:
        raise ValueError(f"{label} has no conformer")
    return m


def coords(m):
    if m is None:
        raise ValueError("coords() received None")
    if m.GetNumConformers() == 0:
        raise ValueError("molecule has no conformer")
    conf = m.GetConformer()
    return np.array([[conf.GetAtomPosition(i).x,
                      conf.GetAtomPosition(i).y,
                      conf.GetAtomPosition(i).z] for i in range(m.GetNumAtoms())], dtype=float)


def set_coords(m, xyz):
    if m is None:
        raise ValueError("set_coords() received None")
    if xyz.shape != (m.GetNumAtoms(), 3):
        raise ValueError("Coordinate array shape does not match molecule")
    if m.GetNumConformers() == 0:
        raise ValueError("molecule has no conformer")
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        x, y, z = map(float, xyz[i])
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))


def apply_T_xyz(xyz, T):
    R = T[:3, :3]
    t = T[:3, 3]
    return xyz @ R.T + t


def write_mol_no_kekulize(m, path, forceV3000=False):
    block = Chem.MolToMolBlock(m, kekulize=False, forceV3000=forceV3000)
    with open(path, "w", encoding="utf-8") as f:
        f.write(block)


def heavy_atom_count(m):
    if m is None:
        return 0
    return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1)


def ensure_ringinfo(m):
    if m is None:
        return None
    try:
        m.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    try:
        Chem.GetSymmSSSR(m)
    except Exception:
        try:
            Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
        except Exception:
            pass
    return m


def get_frags_indices(m):
    frags = Chem.GetMolFrags(m, asMols=False, sanitizeFrags=False)
    frags = [list(f) for f in frags]
    sizes = [len(f) for f in frags]
    return frags, sizes


def pick_cd_idx(m):
    frags, sizes = get_frags_indices(m)
    return frags[int(np.argmax(sizes))]


def pick_guest_idx_largest_non_cd(m, min_atoms=3):
    frags, sizes = get_frags_indices(m)
    cd_i = int(np.argmax(sizes))
    cand = []
    for i, idxs in enumerate(frags):
        if i == cd_i:
            continue
        if len(idxs) >= min_atoms:
            cand.append(idxs)
    if not cand:
        return None
    return max(cand, key=len)


def extract_submol_by_atomset(parent, atom_idx_list):
    keep = sorted(set(int(x) for x in atom_idx_list))
    keep_set = set(keep)

    rw = Chem.RWMol(parent)
    to_remove = [a.GetIdx() for a in parent.GetAtoms() if a.GetIdx() not in keep_set]
    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(idx)
    sub = rw.GetMol()

    if parent.GetNumConformers() > 0 and sub.GetNumAtoms() == len(keep):
        conf_parent = parent.GetConformer()
        conf = Chem.Conformer(sub.GetNumAtoms())
        for new_i, old_i in enumerate(keep):
            pos = conf_parent.GetAtomPosition(old_i)
            conf.SetAtomPosition(new_i, pos)
        sub.RemoveAllConformers()
        sub.AddConformer(conf, assignId=True)

    ensure_ringinfo(sub)
    new_to_parent = keep[:]
    return sub, new_to_parent


def extract_submol_heavy_by_atomset(parent, atom_idx_list):
    heavy = [int(i) for i in sorted(set(atom_idx_list))
             if parent.GetAtomWithIdx(int(i)).GetAtomicNum() > 1]
    if not heavy:
        return None, []
    return extract_submol_by_atomset(parent, heavy)


def normalize(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def pca_first_axis(points):
    if points is None:
        return None
    if points.shape[0] < 3:
        return None

    centered = points - points.mean(axis=0)
    C = centered.T @ centered
    vals, vecs = np.linalg.eigh(C)
    axis = vecs[:, int(np.argmax(vals))]

    if np.linalg.norm(axis) < 1e-12:
        return None

    return axis


def angle_deg(u, v):
    if u is None or v is None:
        return np.nan

    u = normalize(u)
    v = normalize(v)

    if np.linalg.norm(u) < 1e-12 or np.linalg.norm(v) < 1e-12:
        return np.nan

    c = float(np.dot(u, v))
    c = max(-1.0, min(1.0, abs(c)))
    return float(np.degrees(np.arccos(c)))


SM_O6 = "[CH2][OX2]"
SM_SEC = "[CH][OX2]"
patt_o6 = Chem.MolFromSmarts(SM_O6)
patt_sec = Chem.MolFromSmarts(SM_SEC)


def matched_atom_set(m, patt):
    ms = m.GetSubstructMatches(patt)
    s = set()
    for mt in ms:
        for i in mt:
            s.add(i)
    return s


def frame_from_rims_on_full_mol(m, cd_idx_list):
    xyz = coords(m)
    cd_set = set(cd_idx_list)

    o6_idx = sorted(matched_atom_set(m, patt_o6).intersection(cd_set))
    sec_idx = sorted(matched_atom_set(m, patt_sec).intersection(cd_set))
    if len(o6_idx) < 3 or len(sec_idx) < 3:
        return None

    o6_com = xyz[o6_idx].mean(axis=0)
    sec_com = xyz[sec_idx].mean(axis=0)

    origin = xyz[cd_idx_list].mean(axis=0)
    z = normalize(sec_com - o6_com)

    rim_pts = xyz[o6_idx] - xyz[o6_idx].mean(axis=0)
    rim_pts = rim_pts - (rim_pts @ z)[:, None] * z[None, :]
    x = pca_first_axis(rim_pts)

    if x is None:
        all_pts = xyz[cd_idx_list] - xyz[cd_idx_list].mean(axis=0)
        all_pts = all_pts - (all_pts @ z)[:, None] * z[None, :]
        x = pca_first_axis(all_pts)

    if x is None:
        return None

    x = normalize(x)
    y = normalize(np.cross(z, x))
    x = normalize(np.cross(y, z))

    B = np.column_stack([x, y, z])
    return origin, B, o6_idx, sec_idx


def score_T_by_rim(m_prb, m_ref, T, cd_idx_prb, cd_idx_ref):
    xyz_ref = coords(m_ref)
    xyz_prb = coords(m_prb)
    xyz_prb_t = apply_T_xyz(xyz_prb, T)

    cd_set_ref = set(cd_idx_ref)
    cd_set_prb = set(cd_idx_prb)

    o6_ref = sorted(matched_atom_set(m_ref, patt_o6).intersection(cd_set_ref))
    o6_prb = sorted(matched_atom_set(m_prb, patt_o6).intersection(cd_set_prb))
    sec_ref = sorted(matched_atom_set(m_ref, patt_sec).intersection(cd_set_ref))

    if len(o6_ref) < 3 or len(o6_prb) < 3 or len(sec_ref) < 3:
        return float("inf")

    o6_ref_com = xyz_ref[o6_ref].mean(axis=0)
    sec_ref_com = xyz_ref[sec_ref].mean(axis=0)

    axis = normalize(sec_ref_com - o6_ref_com)
    center = o6_ref_com

    A = xyz_ref[o6_ref]
    B = xyz_prb_t[o6_prb]

    def to_plane_and_angle(P):
        P2 = P - center
        P2 = P2 - (P2 @ axis)[:, None] * axis[None, :]
        v0 = P2[0]
        if np.linalg.norm(v0) < 1e-12 and P2.shape[0] > 1:
            v0 = P2[1]
        ex = v0 / (np.linalg.norm(v0) + 1e-12)
        ey = np.cross(axis, ex)
        ang = np.arctan2(P2 @ ey, P2 @ ex)
        return P2, ang

    A2, angA = to_plane_and_angle(A)
    B2, angB = to_plane_and_angle(B)

    A2 = A2[np.argsort(angA)]
    B2 = B2[np.argsort(angB)]

    n = min(len(A2), len(B2))
    if n < 3:
        return float("inf")

    A2 = A2[:n]
    B2 = B2[:n]

    best = float("inf")
    for k in range(n):
        Bk = np.roll(B2, k, axis=0)
        d2 = np.sum((A2 - Bk) ** 2, axis=1)
        rms = float(np.sqrt(np.mean(d2)))
        if rms < best:
            best = rms
    return best


def build_T_axis_frame(m_prb, m_ref, cd_idx_prb, cd_idx_ref):
    prb_frame = frame_from_rims_on_full_mol(m_prb, cd_idx_prb)
    ref_frame = frame_from_rims_on_full_mol(m_ref, cd_idx_ref)
    if prb_frame is None or ref_frame is None:
        return None, "rims_not_found"

    o_prb, B_prb, _, _ = prb_frame
    o_ref, B_ref, _, _ = ref_frame

    R1 = B_ref @ B_prb.T
    t1 = o_ref - (o_prb @ R1.T)
    T1 = np.eye(4, dtype=float)
    T1[:3, :3] = R1
    T1[:3, 3] = t1

    B_prb2 = B_prb.copy()
    B_prb2[:, 0] *= -1
    B_prb2[:, 1] *= -1

    R2 = B_ref @ B_prb2.T
    t2 = o_ref - (o_prb @ R2.T)
    T2 = np.eye(4, dtype=float)
    T2[:3, :3] = R2
    T2[:3, 3] = t2

    s1 = score_T_by_rim(m_prb, m_ref, T1, cd_idx_prb, cd_idx_ref)
    s2 = score_T_by_rim(m_prb, m_ref, T2, cd_idx_prb, cd_idx_ref)

    if s2 < s1:
        return T2, "ok_axis_frame_180"
    return T1, "ok_axis_frame"


def cd_axis_vector(m_full, cd_idx):
    xyz = coords(m_full)
    cd_set = set(cd_idx)
    o6_idx = sorted(matched_atom_set(m_full, patt_o6).intersection(cd_set))
    sec_idx = sorted(matched_atom_set(m_full, patt_sec).intersection(cd_set))
    if len(o6_idx) < 3 or len(sec_idx) < 3:
        return None, None, None, None, None
    o6_com = xyz[o6_idx].mean(axis=0)
    sec_com = xyz[sec_idx].mean(axis=0)
    v = sec_com - o6_com
    return o6_com, sec_com, v, o6_idx, sec_idx


def cd_mid_and_z_to_o6(m_full, cd_idx):
    o6_com, sec_com, v, _, _ = cd_axis_vector(m_full, cd_idx)
    if v is None:
        return None, None, None, None
    mid = 0.5 * (o6_com + sec_com)
    z_to_o6 = normalize(o6_com - mid)
    return o6_com, sec_com, mid, z_to_o6


def flip_in_reference_frame(xyz, origin_ref, B_ref):
    Rx180 = np.array([[1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0]], dtype=float)
    Rflip = B_ref @ Rx180 @ B_ref.T
    p = xyz - origin_ref[None, :]
    p2 = p @ Rflip.T
    return p2 + origin_ref[None, :]


def find_mcs_strict(ref, prb, timeout=12):
    return rdFMCS.FindMCS(
        [ref, prb],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        matchValences=False,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        timeout=timeout
    )


def find_mcs_soft(ref, prb, timeout=12):
    return rdFMCS.FindMCS(
        [ref, prb],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        matchValences=False,
        ringMatchesRingOnly=False,
        completeRingsOnly=False,
        timeout=timeout
    )


def _parent_label(parent_mol, parent_idx):
    a = parent_mol.GetAtomWithIdx(int(parent_idx))
    return f"{a.GetSymbol()}{int(parent_idx) + 1}"


def kabsch_transform(P, Q):
    if P.shape != Q.shape or P.shape[0] < 3:
        return None, np.nan

    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)

    P0 = P - Pc
    Q0 = Q - Qc

    H = Q0.T @ P0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = Pc - (Qc @ R.T)

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t

    Q_fit = apply_T_xyz(Q, T)
    rmsd = float(np.sqrt(np.mean(np.sum((P - Q_fit) ** 2, axis=1))))
    return T, rmsd


def get_cd_core_idx(m, cd_idx):
    cd_set = set(cd_idx)
    core = []
    ring_info = m.GetRingInfo()
    ring_atom_flags = set()
    try:
        for ring in ring_info.AtomRings():
            for a in ring:
                ring_atom_flags.add(a)
    except Exception:
        pass

    for i in sorted(cd_set):
        atom = m.GetAtomWithIdx(i)
        if atom.GetAtomicNum() <= 1:
            continue
        if i not in ring_atom_flags:
            continue
        if atom.GetAtomicNum() not in (6, 8):
            continue
        core.append(i)

    return core


def _best_mapping_for_host_fit(ref_host_heavy, prb_host_heavy, patt, eps=1e-6):
    ensure_ringinfo(ref_host_heavy)
    ensure_ringinfo(prb_host_heavy)

    ref_matches = list(ref_host_heavy.GetSubstructMatches(patt, uniquify=False))
    prb_matches = list(prb_host_heavy.GetSubstructMatches(patt, uniquify=False))

    n_ref = len(ref_matches)
    n_prb = len(prb_matches)
    total_pairings = n_ref * n_prb

    if n_ref == 0 or n_prb == 0:
        return None, None, np.nan, n_ref, n_prb, total_pairings

    ref_xyz = coords(ref_host_heavy)
    prb_xyz = coords(prb_host_heavy)

    best_rm = None
    best_pm = None
    best_r = None

    for rm in ref_matches:
        rm = tuple(rm)
        P = ref_xyz[list(rm)]

        for pm in prb_matches:
            pm = tuple(pm)

            ok = True
            for a, b in zip(rm, pm):
                if ref_host_heavy.GetAtomWithIdx(a).GetAtomicNum() != prb_host_heavy.GetAtomWithIdx(b).GetAtomicNum():
                    ok = False
                    break
            if not ok:
                continue

            Q = prb_xyz[list(pm)]
            _, r = kabsch_transform(P, Q)
            if np.isnan(r):
                continue

            if best_r is None or r < best_r - eps:
                best_r = r
                best_rm = rm
                best_pm = pm

    return best_rm, best_pm, float(best_r) if best_r is not None else np.nan, n_ref, n_prb, total_pairings


def refine_host_alignment_on_cd_core(ref_full, prb_full, ref_cd_idx, prb_cd_idx, timeout=20, min_atoms=12):
    ref_core_idx = get_cd_core_idx(ref_full, ref_cd_idx)
    prb_core_idx = get_cd_core_idx(prb_full, prb_cd_idx)

    ref_host_heavy, _ = extract_submol_heavy_by_atomset(ref_full, ref_core_idx)
    prb_host_heavy, _ = extract_submol_heavy_by_atomset(prb_full, prb_core_idx)

    if ref_host_heavy is None or prb_host_heavy is None:
        return None, "no_host_core", 0, np.nan, np.nan

    ensure_ringinfo(ref_host_heavy)
    ensure_ringinfo(prb_host_heavy)

    res = find_mcs_strict(ref_host_heavy, prb_host_heavy, timeout=timeout)
    if res is None or getattr(res, "numAtoms", 0) < min_atoms:
        res = find_mcs_soft(ref_host_heavy, prb_host_heavy, timeout=timeout)

    if res is None or getattr(res, "numAtoms", 0) < min_atoms:
        return None, "host_core_mcs_too_small", 0, np.nan, np.nan

    patt = Chem.MolFromSmarts(res.smartsString)
    if patt is None:
        return None, "bad_host_core_mcs_smarts", 0, np.nan, np.nan

    best_rm, best_pm, best_r_before, _, _, _ = _best_mapping_for_host_fit(ref_host_heavy, prb_host_heavy, patt)
    if best_rm is None or best_pm is None:
        return None, "host_core_match_failed", 0, np.nan, np.nan

    ref_xyz = coords(ref_host_heavy)
    prb_xyz = coords(prb_host_heavy)

    P = ref_xyz[list(best_rm)]
    Q = prb_xyz[list(best_pm)]

    T_refine, rmsd_after = kabsch_transform(P, Q)
    if T_refine is None:
        return None, "kabsch_failed", 0, best_r_before, np.nan

    return T_refine, "ok", len(best_rm), best_r_before, rmsd_after


def normalize_guest_for_matching_relaxed(m):
    if m is None:
        return None

    x = Chem.Mol(m)
    ensure_ringinfo(x)

    try:
        Chem.SanitizeMol(x)
    except Exception:
        pass

    rw = Chem.RWMol(x)

    for atom in rw.GetAtoms():
        atom.SetNoImplicit(True)

    for bond in rw.GetBonds():
        bt = bond.GetBondType()
        if bt not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            bond.SetBondType(Chem.BondType.SINGLE)

    x2 = rw.GetMol()

    try:
        Chem.AssignStereochemistry(x2, cleanIt=True, force=False)
    except Exception:
        pass

    return x2


def normalize_guest_for_matching_topology_relaxed(m):
    if m is None:
        return None

    x = Chem.Mol(m)
    ensure_ringinfo(x)

    try:
        Chem.SanitizeMol(x)
    except Exception:
        pass

    rw = Chem.RWMol(x)

    for atom in rw.GetAtoms():
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(False)

    for bond in rw.GetBonds():
        bond.SetIsAromatic(False)
        bond.SetBondType(Chem.BondType.SINGLE)

    x2 = rw.GetMol()

    try:
        Chem.AssignStereochemistry(x2, cleanIt=True, force=False)
    except Exception:
        pass

    return x2


def get_full_matches_no_stereo(query, target):
    params = Chem.SubstructMatchParameters()
    params.useChirality = False
    params.uniquify = False
    return list(target.GetSubstructMatches(query, params))


def atom_local_signature_strict(m, atom_idx):
    a = m.GetAtomWithIdx(int(atom_idx))

    heavy_neighbors = []
    for nb in a.GetNeighbors():
        if nb.GetAtomicNum() > 1:
            bond = m.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx())

            heavy_neighbors.append((
                nb.GetAtomicNum(),
                bool(nb.GetIsAromatic()),
                bool(nb.IsInRing()),
                str(bond.GetBondType()),
                bool(bond.GetIsAromatic())
            ))

    heavy_neighbors = tuple(sorted(heavy_neighbors))

    return (
        a.GetAtomicNum(),
        bool(a.GetIsAromatic()),
        bool(a.IsInRing()),
        int(a.GetFormalCharge()),
        int(a.GetTotalNumHs()),
        len(heavy_neighbors),
        heavy_neighbors,
    )


def atom_local_signature_resonance_relaxed(m, atom_idx):
    a = m.GetAtomWithIdx(int(atom_idx))

    heavy_neighbor_atomic_nums = []
    for nb in a.GetNeighbors():
        if nb.GetAtomicNum() > 1:
            heavy_neighbor_atomic_nums.append(nb.GetAtomicNum())

    heavy_neighbor_atomic_nums = tuple(sorted(heavy_neighbor_atomic_nums))

    return (
    a.GetAtomicNum(),
    bool(a.GetIsAromatic()),
    bool(a.IsInRing()),
    int(a.GetTotalNumHs()),
    len(heavy_neighbor_atomic_nums),
    heavy_neighbor_atomic_nums,
)


def atom_local_signature_topology_relaxed(m, atom_idx):
    a = m.GetAtomWithIdx(int(atom_idx))

    heavy_neighbor_atomic_nums = []
    for nb in a.GetNeighbors():
        if nb.GetAtomicNum() > 1:
            heavy_neighbor_atomic_nums.append(nb.GetAtomicNum())

    heavy_neighbor_atomic_nums = tuple(sorted(heavy_neighbor_atomic_nums))

    return (
        a.GetAtomicNum(),
        bool(a.IsInRing()),
        len(heavy_neighbor_atomic_nums),
        heavy_neighbor_atomic_nums,
    )


def mapping_passes_strict_validation(ref_guest_heavy, prb_guest_heavy, rm, pm):
    if len(rm) != len(pm):
        return False
    for a, b in zip(rm, pm):
        sig_ref = atom_local_signature_strict(ref_guest_heavy, a)
        sig_prb = atom_local_signature_strict(prb_guest_heavy, b)
        if sig_ref != sig_prb:
            return False
    return True


def mapping_passes_resonance_relaxed_validation(ref_guest_heavy, prb_guest_heavy, rm, pm):
    if len(rm) != len(pm):
        return False
    for a, b in zip(rm, pm):
        sig_ref = atom_local_signature_resonance_relaxed(ref_guest_heavy, a)
        sig_prb = atom_local_signature_resonance_relaxed(prb_guest_heavy, b)
        if sig_ref != sig_prb:
            return False
    return True


def mapping_passes_topology_relaxed_validation(ref_guest_heavy, prb_guest_heavy, rm, pm):
    if len(rm) != len(pm):
        return False
    for a, b in zip(rm, pm):
        sig_ref = atom_local_signature_topology_relaxed(ref_guest_heavy, a)
        sig_prb = atom_local_signature_topology_relaxed(prb_guest_heavy, b)
        if sig_ref != sig_prb:
            return False
    return True


def best_mapping_from_candidates(ref_guest_heavy, prb_guest_heavy, candidates, validation_fn, eps=1e-3):
    ref_xyz = coords(ref_guest_heavy)
    prb_xyz = coords(prb_guest_heavy)

    best_rm = None
    best_pm = None
    best_r = None
    second_best = None
    best_count = 0
    valid_count = 0

    for rm, pm in candidates:
        if not validation_fn(ref_guest_heavy, prb_guest_heavy, rm, pm):
            continue

        valid_count += 1
        d2 = []
        ok = True

        for a, b in zip(rm, pm):
            a_ref = ref_guest_heavy.GetAtomWithIdx(a)
            a_prb = prb_guest_heavy.GetAtomWithIdx(b)

            if a_ref.GetAtomicNum() != a_prb.GetAtomicNum():
                ok = False
                break

            vv = ref_xyz[a] - prb_xyz[b]
            d2.append(float(np.dot(vv, vv)))

        if not ok or not d2:
            continue

        r = float(np.sqrt(np.mean(d2)))

        if best_r is None or r < best_r - eps:
            second_best = best_r
            best_r = r
            best_rm = tuple(rm)
            best_pm = tuple(pm)
            best_count = 1
        elif abs(r - best_r) <= eps:
            best_count += 1
        else:
            if second_best is None or r < second_best:
                second_best = r

    if best_rm is None or best_pm is None:
        return None, None, np.nan, False, False, np.nan, valid_count

    is_unique = (best_count == 1)
    is_tie = (best_count > 1)
    delta2 = np.nan if second_best is None else float(second_best - best_r)

    return best_rm, best_pm, float(best_r), is_unique, is_tie, delta2, valid_count


def build_full_match_candidates(query_mol, target_mol):
    if query_mol is None or target_mol is None:
        return []

    qn = query_mol.GetNumAtoms()
    tn = target_mol.GetNumAtoms()

    all_candidates = []

    matches_forward = get_full_matches_no_stereo(query_mol, target_mol)
    for pm in matches_forward:
        if len(pm) != qn:
            continue
        rm = tuple(range(qn))
        all_candidates.append((rm, tuple(pm)))

    matches_reverse = get_full_matches_no_stereo(target_mol, query_mol)
    for rm in matches_reverse:
        if len(rm) != tn:
            continue
        pm = tuple(range(tn))
        all_candidates.append((tuple(rm), pm))

    uniq_candidates = []
    seen = set()
    for rm, pm in all_candidates:
        key = (tuple(rm), tuple(pm))
        if key not in seen:
            seen.add(key)
            uniq_candidates.append((tuple(rm), tuple(pm)))

    return uniq_candidates


def best_full_guest_mapping_by_rmsd(ref_guest_heavy, prb_guest_heavy, eps=1e-3):
    if ref_guest_heavy is None or prb_guest_heavy is None:
        return None, "missing_guest", np.nan, False, False, np.nan, None, None, 0

    ref_n = ref_guest_heavy.GetNumAtoms()
    prb_n = prb_guest_heavy.GetNumAtoms()

    if ref_n != prb_n:
        return None, "different_atom_count", np.nan, False, False, np.nan, None, None, 0

    ref_norm = normalize_guest_for_matching_relaxed(ref_guest_heavy)
    prb_norm = normalize_guest_for_matching_relaxed(prb_guest_heavy)

    ref_topo = normalize_guest_for_matching_topology_relaxed(ref_guest_heavy)
    prb_topo = normalize_guest_for_matching_topology_relaxed(prb_guest_heavy)

    try:
        ref_smiles = Chem.MolToSmiles(ref_norm, canonical=True)
    except Exception:
        ref_smiles = None

    try:
        prb_smiles = Chem.MolToSmiles(prb_norm, canonical=True)
    except Exception:
        prb_smiles = None

    strict_candidates = build_full_match_candidates(ref_norm, prb_norm)
    if strict_candidates:
        best_rm, best_pm, best_r, is_unique, is_tie, delta2, valid_count = best_mapping_from_candidates(
            ref_guest_heavy, prb_guest_heavy, strict_candidates, mapping_passes_strict_validation, eps=eps
        )
        if best_rm is not None:
            return (best_rm, best_pm), "full_match_strict", best_r, is_unique, is_tie, delta2, ref_smiles, prb_smiles, valid_count

        best_rm, best_pm, best_r, is_unique, is_tie, delta2, valid_count = best_mapping_from_candidates(
            ref_guest_heavy, prb_guest_heavy, strict_candidates, mapping_passes_resonance_relaxed_validation, eps=eps
        )
        if best_rm is not None:
            return (best_rm, best_pm), "full_match_resonance_relaxed", best_r, is_unique, is_tie, delta2, ref_smiles, prb_smiles, valid_count

    topo_candidates = build_full_match_candidates(ref_topo, prb_topo)
    if topo_candidates:
        best_rm, best_pm, best_r, is_unique, is_tie, delta2, valid_count = best_mapping_from_candidates(
            ref_guest_heavy, prb_guest_heavy, topo_candidates, mapping_passes_topology_relaxed_validation, eps=eps
        )
        if best_rm is not None:
            return (best_rm, best_pm), "full_match_topology_relaxed", best_r, is_unique, is_tie, delta2, ref_smiles, prb_smiles, valid_count

    n_candidates = max(len(strict_candidates), len(topo_candidates))
    return None, "no_valid_full_match", np.nan, False, False, np.nan, ref_smiles, prb_smiles, n_candidates


def _best_mapping_min_rmsd_element_safe(ref_guest_heavy, prb_guest_heavy, patt, eps=1e-3):
    ensure_ringinfo(ref_guest_heavy)
    ensure_ringinfo(prb_guest_heavy)

    ref_matches = list(ref_guest_heavy.GetSubstructMatches(patt, uniquify=False))
    prb_matches = list(prb_guest_heavy.GetSubstructMatches(patt, uniquify=False))

    n_ref = len(ref_matches)
    n_prb = len(prb_matches)
    total_pairings = n_ref * n_prb
    if n_ref == 0 or n_prb == 0:
        return None, None, np.nan, n_ref, n_prb, total_pairings, False, False, np.nan

    ref_xyz = coords(ref_guest_heavy)
    prb_xyz = coords(prb_guest_heavy)

    best_r = None
    second_best = None
    best_rm = None
    best_pm = None
    best_count = 0

    for rm in ref_matches:
        rm = tuple(rm)
        for pm in prb_matches:
            pm = tuple(pm)

            ok = True
            for a, b in zip(rm, pm):
                if ref_guest_heavy.GetAtomWithIdx(a).GetAtomicNum() != prb_guest_heavy.GetAtomWithIdx(b).GetAtomicNum():
                    ok = False
                    break
            if not ok:
                continue

            d2 = []
            for a, b in zip(rm, pm):
                vv = ref_xyz[a] - prb_xyz[b]
                d2.append(float(np.dot(vv, vv)))
            if not d2:
                continue
            r = float(np.sqrt(np.mean(d2)))

            if best_r is None or r < best_r - eps:
                second_best = best_r
                best_r = r
                best_rm = rm
                best_pm = pm
                best_count = 1
            elif abs(r - best_r) <= eps:
                best_count += 1
            else:
                if second_best is None or r < second_best:
                    second_best = r

    if best_rm is None or best_pm is None:
        return None, None, np.nan, n_ref, n_prb, total_pairings, False, False, np.nan

    is_unique = (best_count == 1)
    is_tie = (best_count > 1)
    delta2 = np.nan
    if second_best is not None and best_r is not None:
        delta2 = float(second_best - best_r)

    return best_rm, best_pm, float(best_r), n_ref, n_prb, total_pairings, is_unique, is_tie, delta2


def guest_rmsd_and_orientation_heavy(ref_full, prb_full,
                                     ref_guest_idx, prb_guest_idx,
                                     ref_o6_com_full, ref_zhat,
                                     timeout=12):
    if prb_guest_idx is None:
        return (np.nan, "no_guest", 0,
                np.nan, "no_guest", np.nan, np.nan, 0,
                [], "", 0, 0, 0, False, False, np.nan, "none")

    ref_guest_heavy, ref_map = extract_submol_heavy_by_atomset(ref_full, ref_guest_idx)
    prb_guest_heavy, prb_map = extract_submol_heavy_by_atomset(prb_full, prb_guest_idx)

    if ref_guest_heavy is None or prb_guest_heavy is None:
        return (np.nan, "no_heavy_guest", 0,
                np.nan, "no_heavy_guest", np.nan, np.nan, 0,
                [], "", 0, 0, 0, False, False, np.nan, "none")

    ensure_ringinfo(ref_guest_heavy)
    ensure_ringinfo(prb_guest_heavy)

    match_mode = "full_guest"

    full_map, full_status, full_rmsd, full_unique, full_tie, full_delta2, _, _, full_n_matches = \
        best_full_guest_mapping_by_rmsd(ref_guest_heavy, prb_guest_heavy)

    if full_map is not None:
        best_rm, best_pm = full_map
        best_rmsd = full_rmsd
        n_ref = 1
        n_prb = full_n_matches
        total_pairings = full_n_matches
        is_unique = full_unique
        is_tie = full_tie
        delta2 = full_delta2

        if full_status == "full_match_strict":
            match_mode = "full_guest_strict"
        elif full_status == "full_match_resonance_relaxed":
            match_mode = "full_guest_resonance_relaxed"
        elif full_status == "full_match_topology_relaxed":
            match_mode = "full_guest_topology_relaxed"
        else:
            match_mode = "full_guest"
    else:
        match_mode = "mcs"

        res = find_mcs_strict(ref_guest_heavy, prb_guest_heavy, timeout=timeout)
        if res is None or getattr(res, "numAtoms", 0) < 4:
            res = find_mcs_soft(ref_guest_heavy, prb_guest_heavy, timeout=timeout)

        if res is None or getattr(res, "numAtoms", 0) < 4:
            return (np.nan, "no_mcs", 0,
                    np.nan, "no_mcs", np.nan, np.nan, 0,
                    [], "", 0, 0, 0, False, False, np.nan, match_mode)

        patt = Chem.MolFromSmarts(res.smartsString)
        if patt is None:
            return (np.nan, "bad_mcs_smarts", 0,
                    np.nan, "bad_mcs_smarts", np.nan, np.nan, 0,
                    [], "", 0, 0, 0, False, False, np.nan, match_mode)

        best_rm, best_pm, best_rmsd, n_ref, n_prb, total_pairings, is_unique, is_tie, delta2 = \
            _best_mapping_min_rmsd_element_safe(ref_guest_heavy, prb_guest_heavy, patt)

        if best_rm is None or best_pm is None or len(best_rm) < 4:
            return (np.nan, "mcs_match_failed", 0,
                    np.nan, "mcs_match_failed", np.nan, np.nan, 0,
                    [], "", n_ref, n_prb, total_pairings, is_unique, is_tie, delta2, match_mode)

    mapping_rows = []
    for (r_sub, p_sub) in zip(best_rm, best_pm):
        r_parent = ref_map[int(r_sub)]
        p_parent = prb_map[int(p_sub)]
        mapping_rows.append({
            "atom_label_tested": _parent_label(prb_full, p_parent),
            "atom_label_reference": _parent_label(ref_full, r_parent),
        })

    xyz_ref = coords(ref_guest_heavy)
    xyz_prb = coords(prb_guest_heavy)

    best_i = None
    best_j = None
    best_span_ref = -1.0

    for ii, jj in itertools.combinations(range(len(best_rm)), 2):
        a = best_rm[ii]
        b = best_rm[jj]
        d = float(np.linalg.norm(xyz_ref[a] - xyz_ref[b]))
        if d > best_span_ref:
            best_span_ref = d
            best_i = ii
            best_j = jj

    pair_text = ""
    if best_i is not None:
        rA_parent = ref_map[int(best_rm[best_i])]
        pA_parent = prb_map[int(best_pm[best_i])]
        rB_parent = ref_map[int(best_rm[best_j])]
        pB_parent = prb_map[int(best_pm[best_j])]
        pair_text = f"A: {_parent_label(prb_full, pA_parent)}->{_parent_label(ref_full, rA_parent)} | B: {_parent_label(prb_full, pB_parent)}->{_parent_label(ref_full, rB_parent)}"

    if best_i is None:
        return (best_rmsd, "ok", len(best_rm),
                np.nan, "no_endpoints", np.nan, np.nan, len(best_rm),
                mapping_rows, pair_text, n_ref, n_prb, total_pairings, is_unique, is_tie, delta2, match_mode)

    ref_a = best_rm[best_i]
    ref_b = best_rm[best_j]
    prb_a = best_pm[best_i]
    prb_b = best_pm[best_j]

    probe_distance = float(np.linalg.norm(xyz_prb[prb_a] - xyz_prb[prb_b]))

    ref_pa = float(np.dot(xyz_ref[ref_a] - ref_o6_com_full, ref_zhat))
    ref_pb = float(np.dot(xyz_ref[ref_b] - ref_o6_com_full, ref_zhat))
    up_is_a = ref_pa > ref_pb

    prb_pa = float(np.dot(xyz_prb[prb_a] - ref_o6_com_full, ref_zhat))
    prb_pb = float(np.dot(xyz_prb[prb_b] - ref_o6_com_full, ref_zhat))
    prb_up_is_a = prb_pa > prb_pb

    orient = 1 if (prb_up_is_a == up_is_a) else 0

    return (best_rmsd, "ok", len(best_rm),
            orient, "ok", best_span_ref, probe_distance, len(best_rm),
            mapping_rows, pair_text, n_ref, n_prb, total_pairings, is_unique, is_tie, delta2, match_mode)


def build_output_dirs():
    base_temp_dir = os.path.join(
        tempfile.gettempdir(),
        "CDGA_Tool_processed_molfiles"
    )
    os.makedirs(base_temp_dir, exist_ok=True)

    out_dir = os.path.join(base_temp_dir, "CDGA_Align_molfiles")
    os.makedirs(out_dir, exist_ok=True)

    match_dir = os.path.join(out_dir, "Matching_atoms")
    os.makedirs(match_dir, exist_ok=True)

    return out_dir, match_dir


def prepare_reference(mols):
    if len(mols) == 0:
        raise ValueError("Input table is empty")

    ref = validate_mol_has_3d(mols[0], "Reference molecule")
    ref = Chem.Mol(ref)
    validate_mol_has_3d(ref, "Reference molecule")
    ensure_ringinfo(ref)

    ref_cd_idx = pick_cd_idx(ref)
    ref_frame = frame_from_rims_on_full_mol(ref, ref_cd_idx)
    if ref_frame is None:
        raise ValueError("CD rims not found in reference structure")
    ref_origin, ref_B, _, _ = ref_frame

    ref_o6_com, ref_sec_com, ref_v, _, _ = cd_axis_vector(ref, ref_cd_idx)
    if ref_v is None:
        raise ValueError("Narrow to wide axis could not be calculated for reference structure")
    ref_zhat = normalize(ref_v)

    _, _, ref_mid, ref_z_to_o6 = cd_mid_and_z_to_o6(ref, ref_cd_idx)
    if ref_z_to_o6 is None:
        raise ValueError("Midpoint and O6 direction axis could not be calculated for reference structure")

    ref_guest_idx = pick_guest_idx_largest_non_cd(ref, min_atoms=MIN_GUEST_ATOMS)
    if ref_guest_idx is None:
        raise ValueError("Guest molecule not found in reference structure")

    ref_guest_heavy, _ = extract_submol_heavy_by_atomset(ref, ref_guest_idx)
    ref_guest_atoms_heavy = heavy_atom_count(ref_guest_heavy) if ref_guest_heavy is not None else 0

    try:
        ref_guest_smiles_original_value = Chem.MolToSmiles(ref_guest_heavy, canonical=True)
    except Exception:
        ref_guest_smiles_original_value = np.nan

    try:
        ref_guest_smiles_matching_value = Chem.MolToSmiles(
            normalize_guest_for_matching_topology_relaxed(ref_guest_heavy),
            canonical=True
        )
    except Exception:
        ref_guest_smiles_matching_value = np.nan

    return {
        "ref": ref,
        "ref_cd_idx": ref_cd_idx,
        "ref_origin": ref_origin,
        "ref_B": ref_B,
        "ref_o6_com": ref_o6_com,
        "ref_v": ref_v,
        "ref_zhat": ref_zhat,
        "ref_mid": ref_mid,
        "ref_z_to_o6": ref_z_to_o6,
        "ref_guest_idx": ref_guest_idx,
        "ref_guest_atoms_heavy_noH": ref_guest_atoms_heavy,
        "ref_guest_smiles_original_value": ref_guest_smiles_original_value,
        "ref_guest_smiles_matching_value": ref_guest_smiles_matching_value,
    }


def compute_fragment_metrics(m_full):
    frags, sizes = get_frags_indices(m_full)
    cd_idx = pick_cd_idx(m_full)
    guest_idx = pick_guest_idx_largest_non_cd(m_full, min_atoms=MIN_GUEST_ATOMS)

    host_sub, _ = extract_submol_heavy_by_atomset(m_full, cd_idx)
    core_idx = get_cd_core_idx(m_full, cd_idx)
    core_sub, _ = extract_submol_heavy_by_atomset(m_full, core_idx)

    if guest_idx is None:
        guest_atoms_total = 0
        guest_atoms_heavy_noH = 0
        g_heavy = None
    else:
        guest_atoms_total = len(guest_idx)
        g_heavy, _ = extract_submol_heavy_by_atomset(m_full, guest_idx)
        guest_atoms_heavy_noH = heavy_atom_count(g_heavy) if g_heavy is not None else 0

    return {
        "frags": frags,
        "frag_sizes": sizes,
        "cd_idx": cd_idx,
        "guest_idx": guest_idx,
        "host_atoms_total": len(cd_idx),
        "host_atoms_heavy_noH": heavy_atom_count(host_sub),
        "host_core_atoms_heavy_noH": heavy_atom_count(core_sub),
        "guest_atoms_total": guest_atoms_total,
        "guest_atoms_heavy_noH": guest_atoms_heavy_noH,
        "guest_heavy_mol": g_heavy,
    }


def compute_guest_smiles_debug(g_heavy, ref_guest_smiles_original_value, ref_guest_smiles_matching_value):
    if g_heavy is None:
        return np.nan, np.nan

    try:
        prb_original_smiles = Chem.MolToSmiles(g_heavy, canonical=True)
    except Exception:
        prb_original_smiles = np.nan

    try:
        prb_matching_smiles = Chem.MolToSmiles(
            normalize_guest_for_matching_topology_relaxed(g_heavy),
            canonical=True
        )
    except Exception:
        prb_matching_smiles = np.nan

    return (
        f"REF={ref_guest_smiles_original_value} | PRB={prb_original_smiles}",
        f"REF={ref_guest_smiles_matching_value} | PRB={prb_matching_smiles}"
    )


def align_probe_to_reference(i, m_full, ref_ctx, cd_idx):
    if i == 0:
        T = None
        st = "ref"
    else:
        T, st = build_T_axis_frame(m_full, ref_ctx["ref"], cd_idx, ref_ctx["ref_cd_idx"])

    if T is not None:
        xyz = coords(m_full)
        xyz_t = apply_T_xyz(xyz, T)
        set_coords(m_full, xyz_t)

    did_flip = False
    _, _, prb_v, _, _ = cd_axis_vector(m_full, cd_idx)
    if prb_v is not None and float(np.dot(ref_ctx["ref_v"], prb_v)) < 0:
        xyz2 = coords(m_full)
        xyz2 = flip_in_reference_frame(xyz2, ref_ctx["ref_origin"], ref_ctx["ref_B"])
        set_coords(m_full, xyz2)
        did_flip = True

    align_status = st + ("_postflip" if did_flip else "_noflip")
    return m_full, align_status, did_flip


def refine_host(i, m_full, ref_ctx, cd_idx):
    if i == 0:
        return {
            "host_refine_status": "ref",
            "host_atoms_used_in_refine": 0,
            "host_rmsd_before_refine": np.nan,
            "host_rmsd_after_refine": np.nan,
        }

    T_refine, refine_status, n_fit_atoms, rmsd_before, rmsd_after = refine_host_alignment_on_cd_core(
        ref_full=ref_ctx["ref"],
        prb_full=m_full,
        ref_cd_idx=ref_ctx["ref_cd_idx"],
        prb_cd_idx=cd_idx,
        timeout=HOST_REFINE_TIMEOUT,
        min_atoms=MIN_HOST_CORE_MCS_ATOMS
    )

    if T_refine is not None:
        xyz3 = coords(m_full)
        xyz3_t = apply_T_xyz(xyz3, T_refine)
        set_coords(m_full, xyz3_t)

    return {
        "host_refine_status": refine_status,
        "host_atoms_used_in_refine": int(n_fit_atoms),
        "host_rmsd_before_refine": rmsd_before,
        "host_rmsd_after_refine": rmsd_after,
    }


def compute_guest_geometry(m_full, cd_idx, guest_idx, ref_ctx):
    _, _, prb_mid, prb_z_to_o6 = cd_mid_and_z_to_o6(m_full, cd_idx)

    if prb_z_to_o6 is None or guest_idx is None:
        return {
            "guest_depth_A_from_midpoint_to_O6": np.nan,
            "guest_depth_A_from_ref_midpoint_to_O6": np.nan,
            "guest_angle_deg_vs_axis": np.nan,
        }

    prb_guest_heavy, _ = extract_submol_heavy_by_atomset(m_full, guest_idx)

    if prb_guest_heavy is None:
        return {
            "guest_depth_A_from_midpoint_to_O6": np.nan,
            "guest_depth_A_from_ref_midpoint_to_O6": np.nan,
            "guest_angle_deg_vs_axis": np.nan,
        }

    xyzg = coords(prb_guest_heavy)
    g_com = xyzg.mean(axis=0)
    g_axis_raw = pca_first_axis(xyzg)

    d_local = float(np.dot(g_com - prb_mid, prb_z_to_o6))
    d_ref = float(np.dot(g_com - ref_ctx["ref_mid"], ref_ctx["ref_z_to_o6"]))

    dot_dir = float(np.dot(prb_z_to_o6, ref_ctx["ref_z_to_o6"]))
    if dot_dir < 0:
        d_local = -d_local

    if g_axis_raw is None:
        angle = np.nan
    else:
        guest_axis = normalize(g_axis_raw)
        angle = angle_deg(guest_axis, prb_z_to_o6)

    return {
        "guest_depth_A_from_midpoint_to_O6": d_local,
        "guest_depth_A_from_ref_midpoint_to_O6": d_ref,
        "guest_angle_deg_vs_axis": angle,
    }


def save_aligned_molfile(m_full, out_dir, rowid, used_names):
    base = safe_filename(rowid)
    name = base
    k = 2
    while name.lower() in used_names:
        name = f"{base}__{k}"
        k += 1
    used_names.add(name.lower())

    out_path = os.path.join(out_dir, f"{name}.mol")
    try:
        write_mol_no_kekulize(m_full, out_path, forceV3000=False)
    except Exception:
        write_mol_no_kekulize(m_full, out_path, forceV3000=True)

    return name, out_path


def build_reference_row_outputs():
    return {
        "guest_rmsd_noH": np.nan,
        "guest_rmsd_status": "ref",
        "guest_atoms_used_in_rmsd": 0,
        "guest_match_mode": "ref",
        "guest_full_match_coverage_percent": np.nan,
        "guest_mcs_coverage_percent": np.nan,
        "guest_orientation": np.nan,
        "guest_orientation_status": "ref",
        "guest_orientation_span_A": np.nan,
        "guest_orientation_distance_A": np.nan,
        "guest_orientation_atom_pair": "ref",
        "guest_orientation_atoms_used": 0,
        "mapping_csv_path": np.nan,
        "mapping_status": "ref",
        "match_ref_matches_count": np.nan,
        "match_prb_matches_count": np.nan,
        "match_total_pairings": np.nan,
        "match_mapping_is_unique": np.nan,
        "match_best_rmsd_is_tie": np.nan,
        "match_best_rmsd_second_best_delta": np.nan,
    }


def build_guest_match_outputs(ref_ctx, m_full, guest_idx, guest_atoms_heavy_noH, match_dir, saved_name):
    (rmsd, rst, nuse,
     orient, ost, spanA, probe_dist, nuse_o,
     rows, pair_text, n_ref, n_prb, tot, is_unique, is_tie, delta2, match_mode) = guest_rmsd_and_orientation_heavy(
        ref_full=ref_ctx["ref"],
        prb_full=m_full,
        ref_guest_idx=ref_ctx["ref_guest_idx"],
        prb_guest_idx=guest_idx,
        ref_o6_com_full=ref_ctx["ref_o6_com"],
        ref_zhat=ref_ctx["ref_zhat"],
        timeout=GUEST_MATCH_TIMEOUT
    )

    if match_mode in (
        "full_guest",
        "full_guest_strict",
        "full_guest_resonance_relaxed",
        "full_guest_topology_relaxed"
    ) and guest_atoms_heavy_noH > 0 and nuse > 0:
        guest_full_match_coverage_percent = 100.0 * float(nuse) / float(guest_atoms_heavy_noH)
        guest_mcs_coverage_percent = np.nan
    elif match_mode == "mcs" and guest_atoms_heavy_noH > 0 and nuse > 0:
        guest_full_match_coverage_percent = np.nan
        guest_mcs_coverage_percent = 100.0 * float(nuse) / float(guest_atoms_heavy_noH)
    else:
        guest_full_match_coverage_percent = np.nan
        guest_mcs_coverage_percent = np.nan

    if rows:
        csv_path = os.path.join(match_dir, f"{saved_name}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        mapping_csv_path = csv_path
        mapping_status = "ok"
    else:
        mapping_csv_path = np.nan
        mapping_status = rst

    return {
        "guest_rmsd_noH": rmsd,
        "guest_rmsd_status": rst,
        "guest_atoms_used_in_rmsd": nuse,
        "guest_match_mode": match_mode,
        "guest_full_match_coverage_percent": guest_full_match_coverage_percent,
        "guest_mcs_coverage_percent": guest_mcs_coverage_percent,
        "guest_orientation": orient,
        "guest_orientation_status": ost,
        "guest_orientation_span_A": spanA,
        "guest_orientation_distance_A": probe_dist,
        "guest_orientation_atom_pair": pair_text if pair_text else rst,
        "guest_orientation_atoms_used": nuse_o,
        "mapping_csv_path": mapping_csv_path,
        "mapping_status": mapping_status,
        "match_ref_matches_count": int(n_ref),
        "match_prb_matches_count": int(n_prb),
        "match_total_pairings": int(tot),
        "match_mapping_is_unique": bool(is_unique),
        "match_best_rmsd_is_tie": bool(is_tie),
        "match_best_rmsd_second_best_delta": delta2,
    }


def determine_overall_status(i, align_status, host_refine_status, guest_rmsd_status, mapping_status):
    if i == 0:
        return "ref"
    if "rims_not_found" in str(align_status):
        return "align_failed"
    if str(host_refine_status) not in ("ok", "ref"):
        return f"host_refine_{host_refine_status}"
    if str(guest_rmsd_status) not in ("ok", "ref"):
        return f"guest_match_{guest_rmsd_status}"
    if str(mapping_status) not in ("ok", "ref"):
        return f"mapping_{mapping_status}"
    return "ok"


def process_row(i, rowid, m, ref_ctx, out_dir, match_dir, used_names):
    validate_mol_has_3d(m, f"Molecule at row {rowid}")
    m_full = Chem.Mol(m)
    validate_mol_has_3d(m_full, f"Molecule at row {rowid}")
    ensure_ringinfo(m_full)

    frag_metrics = compute_fragment_metrics(m_full)

    guest_smiles_original_debug, guest_smiles_matching_debug = compute_guest_smiles_debug(
        frag_metrics["guest_heavy_mol"],
        ref_ctx["ref_guest_smiles_original_value"],
        ref_ctx["ref_guest_smiles_matching_value"]
    )

    m_full, align_status, post_flip_applied = align_probe_to_reference(
        i=i,
        m_full=m_full,
        ref_ctx=ref_ctx,
        cd_idx=frag_metrics["cd_idx"]
    )

    host_refine = refine_host(
        i=i,
        m_full=m_full,
        ref_ctx=ref_ctx,
        cd_idx=frag_metrics["cd_idx"]
    )

    guest_geom = compute_guest_geometry(
        m_full=m_full,
        cd_idx=frag_metrics["cd_idx"],
        guest_idx=frag_metrics["guest_idx"],
        ref_ctx=ref_ctx
    )

    saved_name, saved_mol_path = save_aligned_molfile(
        m_full=m_full,
        out_dir=out_dir,
        rowid=rowid,
        used_names=used_names
    )

    if i == 0:
        guest_match = build_reference_row_outputs()
    else:
        guest_match = build_guest_match_outputs(
            ref_ctx=ref_ctx,
            m_full=m_full,
            guest_idx=frag_metrics["guest_idx"],
            guest_atoms_heavy_noH=frag_metrics["guest_atoms_heavy_noH"],
            match_dir=match_dir,
            saved_name=saved_name
        )

    overall_status = determine_overall_status(
        i=i,
        align_status=align_status,
        host_refine_status=host_refine["host_refine_status"],
        guest_rmsd_status=guest_match["guest_rmsd_status"],
        mapping_status=guest_match["mapping_status"]
    )

    rec = {
        "entry_name": str(rowid),
        "saved_mol_path": saved_mol_path,
        "mapping_csv_path": guest_match["mapping_csv_path"],
        "mapping_status": guest_match["mapping_status"],
        "overall_status": overall_status,

        "align_status": align_status,
        "post_flip_applied": post_flip_applied,

        "host_refine_status": host_refine["host_refine_status"],
        "host_atoms_used_in_refine": host_refine["host_atoms_used_in_refine"],
        "host_rmsd_before_refine": host_refine["host_rmsd_before_refine"],
        "host_rmsd_after_refine": host_refine["host_rmsd_after_refine"],

        "n_frags_in_input": len(frag_metrics["frags"]),
        "frag_sizes_in_input": ",".join(str(x) for x in frag_metrics["frag_sizes"]),

        "host_atoms_total": frag_metrics["host_atoms_total"],
        "host_atoms_heavy_noH": frag_metrics["host_atoms_heavy_noH"],
        "host_core_atoms_heavy_noH": frag_metrics["host_core_atoms_heavy_noH"],

        "guest_atoms_total": frag_metrics["guest_atoms_total"],
        "guest_atoms_heavy_noH": frag_metrics["guest_atoms_heavy_noH"],

        "guest_depth_A_from_midpoint_to_O6": guest_geom["guest_depth_A_from_midpoint_to_O6"],
        "guest_depth_A_from_ref_midpoint_to_O6": guest_geom["guest_depth_A_from_ref_midpoint_to_O6"],
        "guest_angle_deg_vs_axis": guest_geom["guest_angle_deg_vs_axis"],

        "guest_rmsd_noH": guest_match["guest_rmsd_noH"],
        "guest_rmsd_status": guest_match["guest_rmsd_status"],
        "guest_atoms_used_in_rmsd": guest_match["guest_atoms_used_in_rmsd"],
        "guest_match_mode": guest_match["guest_match_mode"],
        "guest_full_match_coverage_percent": guest_match["guest_full_match_coverage_percent"],
        "guest_mcs_coverage_percent": guest_match["guest_mcs_coverage_percent"],

        "guest_orientation": guest_match["guest_orientation"],
        "guest_orientation_status": guest_match["guest_orientation_status"],
        "guest_orientation_span_A": guest_match["guest_orientation_span_A"],
        "guest_orientation_distance_A": guest_match["guest_orientation_distance_A"],
        "guest_orientation_atom_pair": guest_match["guest_orientation_atom_pair"],
        "guest_orientation_atoms_used": guest_match["guest_orientation_atoms_used"],

        "match_ref_matches_count": guest_match["match_ref_matches_count"],
        "match_prb_matches_count": guest_match["match_prb_matches_count"],
        "match_total_pairings": guest_match["match_total_pairings"],
        "match_mapping_is_unique": guest_match["match_mapping_is_unique"],
        "match_best_rmsd_is_tie": guest_match["match_best_rmsd_is_tie"],
        "match_best_rmsd_second_best_delta": guest_match["match_best_rmsd_second_best_delta"],

        "guest_smiles_original_debug": guest_smiles_original_debug,
        "guest_smiles_matching_debug": guest_smiles_matching_debug,

        "guest_ref_heavy_noH": ref_ctx["ref_guest_atoms_heavy_noH"],
    }

    return rec


out_dir, match_dir = build_output_dirs()
ref_ctx = prepare_reference(mols)

records = []
used_names = set()

for i, (rowid, m) in enumerate(zip(df.index.astype(str).tolist(), mols)):
    rec = process_row(
        i=i,
        rowid=rowid,
        m=m,
        ref_ctx=ref_ctx,
        out_dir=out_dir,
        match_dir=match_dir,
        used_names=used_names
    )
    records.append(rec)

out = pd.DataFrame(records)
knio.output_tables[0] = knio.Table.from_pandas(out)
## ============================================================
#
#
## =======================================
# HOST CHARACTERIZATION APPENDED MODULE
## =======================================
# It creates:
#
#   CDGA_Align_molfiles/
#       Host_characterization/
#           [molecule_name]_host_pyranose.xyz
#           host_torsion_angles.csv
#
# Supported cyclodextrins:
#
#   alpha: 6 pyranose units, 36 XYZ atoms
#   beta:  7 pyranose units, 42 XYZ atoms
#   gamma: 8 pyranose units, 48 XYZ atoms
#
# The number of pyranose units is detected automatically from
# the reference host. Every tested host is required to contain
# the same number of pyranose units as the reference.
#
# Pyranose XYZ order for every unit:
#
#   O_ring, C1, C2, C3, C4, C5
#
# Tested structures follow the pyranose order established by
# the reference host and the host core correspondence selected
# by the same MCS plus Kabsch logic used in host refinement.
#
# Glycosidic torsions for linkage i -> i+1:
#
#   phi_i = H1(i), C1(i), O4(i+1), C4(i+1)
#   psi_i = C1(i), O4(i+1), C4(i+1), H4(i+1)
#
# The last linkage closes the cyclodextrin ring:
#
#   phi_N = H1(N), C1(N), O4(1), C4(1)
#   psi_N = C1(N), O4(1), C4(1), H4(1)
#
# Explicit H1 and H4 atoms must be present in the MOL files.
# If a required H atom is absent, the atom label is written as
# NA and the corresponding torsion value is left blank.
# ============================================================
#
#
import csv
from rdkit.Chem import rdMolTransforms


SUPPORTED_PYRANOSE_UNITS = (6, 7, 8)


def _hostchar_load_saved_mol(path):
    m = Chem.MolFromMolFile(
        path,
        sanitize=False,
        removeHs=False,
        strictParsing=False,
    )

    if m is None:
        raise ValueError(f"Could not read aligned MOL file: {path}")

    m.UpdatePropertyCache(strict=False)
    ensure_ringinfo(m)
    validate_mol_has_3d(m, f"Aligned MOL file {path}")

    return m


def _hostchar_pyranose_from_six_membered_ring(m, ring, cd_set):
    ring = tuple(int(i) for i in ring)
    ring_set = set(ring)

    if len(ring) != 6:
        return None

    if not ring_set.issubset(cd_set):
        return None

    oxygens = [
        i for i in ring
        if m.GetAtomWithIdx(i).GetAtomicNum() == 8
    ]

    carbons = [
        i for i in ring
        if m.GetAtomWithIdx(i).GetAtomicNum() == 6
    ]

    if len(oxygens) != 1 or len(carbons) != 5:
        return None

    o_ring = int(oxygens[0])

    o_ring_carbon_neighbors = [
        int(nb.GetIdx())
        for nb in m.GetAtomWithIdx(o_ring).GetNeighbors()
        if nb.GetIdx() in ring_set
        and nb.GetAtomicNum() == 6
    ]

    if len(o_ring_carbon_neighbors) != 2:
        return None

    def has_exocyclic_carbon(c_idx):
        for nb in m.GetAtomWithIdx(int(c_idx)).GetNeighbors():
            j = int(nb.GetIdx())

            if j in ring_set:
                continue

            if j not in cd_set:
                continue

            if nb.GetAtomicNum() == 6:
                return True

        return False

    c5_candidates = [
        c
        for c in o_ring_carbon_neighbors
        if has_exocyclic_carbon(c)
    ]

    if len(c5_candidates) != 1:
        return None

    c5 = int(c5_candidates[0])

    c1 = (
        int(o_ring_carbon_neighbors[0])
        if int(o_ring_carbon_neighbors[1]) == c5
        else int(o_ring_carbon_neighbors[1])
    )

    chain = [c1]
    prev = o_ring
    current = c1

    while current != c5 and len(chain) < 5:
        next_candidates = [
            int(nb.GetIdx())
            for nb in m.GetAtomWithIdx(current).GetNeighbors()
            if nb.GetIdx() in ring_set
            and nb.GetIdx() != prev
            and nb.GetIdx() != o_ring
            and nb.GetAtomicNum() == 6
        ]

        if len(next_candidates) != 1:
            return None

        nxt = int(next_candidates[0])
        chain.append(nxt)
        prev, current = current, nxt

    if len(chain) != 5 or chain[-1] != c5:
        return None

    c1, c2, c3, c4, c5 = [int(x) for x in chain]

    return {
        "O": o_ring,
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "C4": c4,
        "C5": c5,
        "ring_set": set(ring_set),
    }


def _hostchar_find_pyranose_units(
    m,
    cd_idx,
    expected_units=None,
):
    ensure_ringinfo(m)
    cd_set = set(int(i) for i in cd_idx)

    units = []
    seen_rings = set()

    for ring in m.GetRingInfo().AtomRings():
        key = frozenset(int(i) for i in ring)

        if key in seen_rings:
            continue

        unit = _hostchar_pyranose_from_six_membered_ring(
            m,
            ring,
            cd_set,
        )

        if unit is None:
            continue

        seen_rings.add(key)
        units.append(unit)

    if expected_units is None:
        if len(units) not in SUPPORTED_PYRANOSE_UNITS:
            raise ValueError(
                f"Expected an alpha, beta, or gamma cyclodextrin host "
                f"with 6, 7, or 8 pyranose rings, but found {len(units)}"
            )
    elif len(units) != expected_units:
        raise ValueError(
            f"Expected {expected_units} pyranose rings in the host, "
            f"but found {len(units)}"
        )

    return units


def _hostchar_order_reference_pyranoses(m, units):
    c4_to_unit = {
        int(unit["C4"]): i
        for i, unit in enumerate(units)
    }

    next_unit = {}

    for i, unit in enumerate(units):
        c1 = int(unit["C1"])
        ring_set = set(unit["ring_set"])
        candidates = []


        for nb_o in m.GetAtomWithIdx(c1).GetNeighbors():
            o_idx = int(nb_o.GetIdx())

            if o_idx in ring_set:
                continue

            if nb_o.GetAtomicNum() != 8:
                continue

            for nb_c in nb_o.GetNeighbors():
                c_idx = int(nb_c.GetIdx())

                if c_idx == c1:
                    continue

                if c_idx in c4_to_unit:
                    candidates.append(
                        int(c4_to_unit[c_idx])
                    )

        candidates = sorted(set(candidates))

        if len(candidates) != 1:
            raise ValueError(
                f"Could not determine a unique glycosidic successor "
                f"for pyranose unit {i + 1}"
            )

        next_unit[i] = int(candidates[0])

    start = min(
        range(len(units)),
        key=lambda i: int(units[i]["C1"]),
    )

    ordered = []
    seen = set()
    current = int(start)

    for _ in range(len(units)):
        if current in seen:
            raise ValueError(
                "Pyranose ordering closed before all units were visited"
            )

        seen.add(current)
        ordered.append(units[current])
        current = int(next_unit[current])

    if current != start or len(ordered) != len(units):
        raise ValueError(
            "Pyranose units do not form one complete cyclic sequence"
        )

    return ordered


def _hostchar_reference_atom_order(ordered_units):
    ordered_atoms = []

    for unit in ordered_units:
        ordered_atoms.extend([
            int(unit["O"]),
            int(unit["C1"]),
            int(unit["C2"]),
            int(unit["C3"]),
            int(unit["C4"]),
            int(unit["C5"]),
        ])

    return ordered_atoms


def _hostchar_best_host_parent_mapping(
    ref_full,
    prb_full,
    ref_cd_idx,
    prb_cd_idx,
    timeout=20,
    min_atoms=12,
):

    ref_core_idx = get_cd_core_idx(
        ref_full,
        ref_cd_idx,
    )

    prb_core_idx = get_cd_core_idx(
        prb_full,
        prb_cd_idx,
    )

    ref_host_heavy, ref_new_to_parent = (
        extract_submol_heavy_by_atomset(
            ref_full,
            ref_core_idx,
        )
    )

    prb_host_heavy, prb_new_to_parent = (
        extract_submol_heavy_by_atomset(
            prb_full,
            prb_core_idx,
        )
    )

    if ref_host_heavy is None or prb_host_heavy is None:
        return None, "no_host_core", np.nan

    ensure_ringinfo(ref_host_heavy)
    ensure_ringinfo(prb_host_heavy)

    res = find_mcs_strict(
        ref_host_heavy,
        prb_host_heavy,
        timeout=timeout,
    )

    if (
        res is None
        or getattr(res, "numAtoms", 0) < min_atoms
    ):
        res = find_mcs_soft(
            ref_host_heavy,
            prb_host_heavy,
            timeout=timeout,
        )

    if (
        res is None
        or getattr(res, "numAtoms", 0) < min_atoms
    ):
        return None, "host_core_mcs_too_small", np.nan

    patt = Chem.MolFromSmarts(res.smartsString)

    if patt is None:
        return None, "bad_host_core_mcs_smarts", np.nan

    best_rm, best_pm, best_r, _, _, _ = (
        _best_mapping_for_host_fit(
            ref_host_heavy,
            prb_host_heavy,
            patt,
        )
    )

    if best_rm is None or best_pm is None:
        return None, "host_core_match_failed", np.nan

    parent_mapping = {}

    for r_sub, p_sub in zip(best_rm, best_pm):
        r_parent = int(
            ref_new_to_parent[int(r_sub)]
        )

        p_parent = int(
            prb_new_to_parent[int(p_sub)]
        )

        parent_mapping[r_parent] = p_parent

    return parent_mapping, "ok", float(best_r)


def _hostchar_order_probe_pyranoses_from_reference(
    ref_ordered_units,
    prb_units,
    parent_mapping,
):
    ordered_probe = []
    used_probe_indices = set()

    for ref_position, ref_unit in enumerate(
        ref_ordered_units,
        start=1,
    ):
        ref_ring_atoms = [
            int(ref_unit["O"]),
            int(ref_unit["C1"]),
            int(ref_unit["C2"]),
            int(ref_unit["C3"]),
            int(ref_unit["C4"]),
            int(ref_unit["C5"]),
        ]

        missing = [
            idx
            for idx in ref_ring_atoms
            if idx not in parent_mapping
        ]

        if missing:
            missing_text = ", ".join(
                str(int(idx) + 1)
                for idx in missing
            )

            raise ValueError(
                f"Host mapping does not contain all ring atoms for "
                f"reference pyranose {ref_position}. "
                f"Missing atom positions: {missing_text}"
            )

        mapped_ring_set = {
            int(parent_mapping[idx])
            for idx in ref_ring_atoms
        }

        candidates = [
            j
            for j, unit in enumerate(prb_units)
            if set(unit["ring_set"]) == mapped_ring_set
        ]

        if len(candidates) != 1:
            raise ValueError(
                f"Could not assign a unique tested pyranose to "
                f"reference pyranose {ref_position}"
            )

        j = int(candidates[0])

        if j in used_probe_indices:
            raise ValueError(
                "The same tested pyranose was assigned more than once"
            )

        used_probe_indices.add(j)
        ordered_probe.append(prb_units[j])

    if len(ordered_probe) != len(ref_ordered_units):
        raise ValueError(
            "The tested pyranose sequence is incomplete"
        )

    return ordered_probe

def _hostchar_write_labeled_xyz(
    m,
    atom_indices,
    out_path,
    comment_line,
):
    xyz = coords(m)

    with open(
        out_path,
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        f.write(f"{len(atom_indices)}\n")
        f.write(f"{comment_line}\n")

        for atom_idx in atom_indices:
            atom_idx = int(atom_idx)

            label = _parent_label(
                m,
                atom_idx,
            )

            x, y, z = xyz[atom_idx]

            f.write(
                f"{label:<8s}"
                f"{x:12.4f}"
                f"{y:12.4f}"
                f"{z:12.4f}\n"
            )


def _hostchar_single_explicit_hydrogen_neighbor(
    m,
    atom_idx,
):
    h_neighbors = [
        int(nb.GetIdx())
        for nb in m.GetAtomWithIdx(
            int(atom_idx)
        ).GetNeighbors()
        if nb.GetAtomicNum() == 1
    ]

    if len(h_neighbors) == 1:
        return int(h_neighbors[0])

    return None


def _hostchar_find_o4_for_pyranose(
    m,
    previous_unit,
    current_unit,
):
    previous_c1 = int(previous_unit["C1"])
    current_c4 = int(current_unit["C4"])

    candidates = []

    for nb in m.GetAtomWithIdx(
        current_c4
    ).GetNeighbors():
        if nb.GetAtomicNum() != 8:
            continue

        o_idx = int(nb.GetIdx())

        bonded_to_previous_c1 = any(
            int(x.GetIdx()) == previous_c1
            for x in nb.GetNeighbors()
        )

        if bonded_to_previous_c1:
            candidates.append(o_idx)

    candidates = sorted(set(candidates))

    if len(candidates) == 1:
        return int(candidates[0])

    return None


def _hostchar_add_h1_o4_h4(
    m,
    ordered_units,
):
    n = len(ordered_units)
    decorated = []

    for i, unit in enumerate(ordered_units):
        previous_unit = ordered_units[
            (i - 1) % n
        ]

        x = dict(unit)

        x["H1"] = (
            _hostchar_single_explicit_hydrogen_neighbor(
                m,
                int(unit["C1"]),
            )
        )

        x["H4"] = (
            _hostchar_single_explicit_hydrogen_neighbor(
                m,
                int(unit["C4"]),
            )
        )

        x["O4"] = _hostchar_find_o4_for_pyranose(
            m,
            previous_unit,
            unit,
        )

        decorated.append(x)

    return decorated


def _hostchar_atom_label_or_na(m, atom_idx):
    if atom_idx is None:
        return "NA"

    return _parent_label(
        m,
        int(atom_idx),
    )


def _hostchar_pyranose_atom_text(m, unit):
    return (
        f"H1={_hostchar_atom_label_or_na(m, unit.get('H1'))}; "
        f"C1={_hostchar_atom_label_or_na(m, unit.get('C1'))}; "
        f"C4={_hostchar_atom_label_or_na(m, unit.get('C4'))}; "
        f"O4={_hostchar_atom_label_or_na(m, unit.get('O4'))}; "
        f"H4={_hostchar_atom_label_or_na(m, unit.get('H4'))}"
    )


def _hostchar_dihedral_deg(
    m,
    a,
    b,
    c,
    d,
):
    atom_indices = [a, b, c, d]

    if any(x is None for x in atom_indices):
        return np.nan

    conf = m.GetConformer()

    try:
        return float(
            rdMolTransforms.GetDihedralDeg(
                conf,
                int(a),
                int(b),
                int(c),
                int(d),
            )
        )
    except Exception:
        return np.nan


def _hostchar_measure_all_linkages(
    m,
    decorated_units,
):
    n = len(decorated_units)
    values = []

    for i in range(n):
        current_unit = decorated_units[i]
        next_unit = decorated_units[
            (i + 1) % n
        ]

        # phi_i:
        #
        # H1(current)
        # C1(current)
        # O4(next)
        # C4(next)
        phi = _hostchar_dihedral_deg(
            m,
            current_unit.get("H1"),
            current_unit.get("C1"),
            next_unit.get("O4"),
            next_unit.get("C4"),
        )

        # psi_i:
        #
        # C1(current)
        # O4(next)
        # C4(next)
        # H4(next)
        psi = _hostchar_dihedral_deg(
            m,
            current_unit.get("C1"),
            next_unit.get("O4"),
            next_unit.get("C4"),
            next_unit.get("H4"),
        )

        values.append((phi, psi))

    return values


def _hostchar_format_angle(value):
    if value is None:
        return ""

    try:
        if np.isnan(float(value)):
            return ""
    except Exception:
        return ""

    return f"{float(value):.4f}"


def _hostchar_empty_torsion_row(
    entry_name,
    expected_units,
):
    row = {
        "entry_name": str(entry_name),
    }

    for i in range(
        1,
        expected_units + 1,
    ):
        row[f"pyranose_{i}_atoms"] = ""
        row[f"phi{i}"] = ""
        row[f"psi{i}"] = ""

    return row


def _hostchar_build_torsion_row(
    entry_name,
    m,
    ordered_units,
    expected_units,
):
    row = _hostchar_empty_torsion_row(
        entry_name,
        expected_units,
    )

    decorated_units = _hostchar_add_h1_o4_h4(
        m,
        ordered_units,
    )

    linkage_angles = (
        _hostchar_measure_all_linkages(
            m,
            decorated_units,
        )
    )

    for i, unit in enumerate(
        decorated_units,
        start=1,
    ):
        phi, psi = linkage_angles[i - 1]

        row[f"pyranose_{i}_atoms"] = (
            _hostchar_pyranose_atom_text(
                m,
                unit,
            )
        )

        row[f"phi{i}"] = (
            _hostchar_format_angle(phi)
        )

        row[f"psi{i}"] = (
            _hostchar_format_angle(psi)
        )

    return row

def export_host_characterization(
    out_dir,
    records,
):
    host_char_dir = os.path.join(
        out_dir,
        "Host_characterization",
    )

    os.makedirs(
        host_char_dir,
        exist_ok=True,
    )

    if not records:
        print(
            "Host characterization: no processed records"
        )
        return {
            "host_characterization_dir": host_char_dir,
            "host_torsion_angles_csv": "",
        }

    ref_saved_path = records[0].get(
        "saved_mol_path",
        "",
    )

    if not ref_saved_path:
        raise ValueError(
            "Reference record does not contain saved_mol_path"
        )

    ref_full = _hostchar_load_saved_mol(
        ref_saved_path
    )

    ref_cd_idx = pick_cd_idx(ref_full)

    ref_units = _hostchar_find_pyranose_units(
        ref_full,
        ref_cd_idx,
        expected_units=None,
    )

    expected_units = len(ref_units)

    cd_type = {
        6: "alpha",
        7: "beta",
        8: "gamma",
    }[expected_units]

    print(
        f"Host characterization detected {cd_type} cyclodextrin "
        f"with {expected_units} pyranose units"
    )

    ref_ordered_units = (
        _hostchar_order_reference_pyranoses(
            ref_full,
            ref_units,
        )
    )

    ref_atom_order = (
        _hostchar_reference_atom_order(
            ref_ordered_units
        )
    )

    expected_atoms = expected_units * 6

    if len(ref_atom_order) != expected_atoms:
        raise ValueError(
            f"Expected {expected_atoms} reference pyranose ring atoms, "
            f"but obtained {len(ref_atom_order)}"
        )

    ref_saved_name = os.path.splitext(
        os.path.basename(ref_saved_path)
    )[0]

    ref_xyz_path = os.path.join(
        host_char_dir,
        f"{ref_saved_name}_host_pyranose.xyz",
    )

    _hostchar_write_labeled_xyz(
        ref_full,
        ref_atom_order,
        ref_xyz_path,
        ref_saved_name,
    )

    print(
        f"Host pyranose XYZ written: "
        f"{ref_xyz_path}"
    )

    torsion_rows = []

    reference_entry_name = records[0].get(
        "entry_name",
        ref_saved_name,
    )

    torsion_rows.append(
        _hostchar_build_torsion_row(
            entry_name=reference_entry_name,
            m=ref_full,
            ordered_units=ref_ordered_units,
            expected_units=expected_units,
        )
    )

    for rec in records[1:]:
        entry_name = rec.get(
            "entry_name",
            "",
        )

        saved_path = rec.get(
            "saved_mol_path",
            "",
        )

        if not saved_path:
            torsion_rows.append(
                _hostchar_empty_torsion_row(
                    entry_name,
                    expected_units,
                )
            )

            print(
                f"Host characterization skipped for "
                f"{entry_name}: missing saved_mol_path"
            )

            continue

        saved_name = os.path.splitext(
            os.path.basename(saved_path)
        )[0]

        try:
            prb_full = _hostchar_load_saved_mol(
                saved_path
            )

            prb_cd_idx = pick_cd_idx(
                prb_full
            )

            prb_units = (
                _hostchar_find_pyranose_units(
                    prb_full,
                    prb_cd_idx,
                    expected_units=expected_units,
                )
            )

            parent_mapping, map_status, map_rmsd = (
                _hostchar_best_host_parent_mapping(
                    ref_full=ref_full,
                    prb_full=prb_full,
                    ref_cd_idx=ref_cd_idx,
                    prb_cd_idx=prb_cd_idx,
                    timeout=HOST_REFINE_TIMEOUT,
                    min_atoms=MIN_HOST_CORE_MCS_ATOMS,
                )
            )

            if parent_mapping is None:
                raise ValueError(map_status)

            missing_ref_atoms = [
                idx
                for idx in ref_atom_order
                if idx not in parent_mapping
            ]

            if missing_ref_atoms:
                missing_labels = ", ".join(
                    _parent_label(
                        ref_full,
                        idx,
                    )
                    for idx in missing_ref_atoms
                )

                raise ValueError(
                    f"Host mapping does not contain all "
                    f"{expected_atoms} reference pyranose ring atoms. "
                    f"Missing: {missing_labels}"
                )

            prb_atom_order = [
                int(parent_mapping[ref_idx])
                for ref_idx in ref_atom_order
            ]

            prb_xyz_path = os.path.join(
                host_char_dir,
                f"{saved_name}_host_pyranose.xyz",
            )

            _hostchar_write_labeled_xyz(
                prb_full,
                prb_atom_order,
                prb_xyz_path,
                saved_name,
            )

            ordered_probe_units = (
                _hostchar_order_probe_pyranoses_from_reference(
                    ref_ordered_units=ref_ordered_units,
                    prb_units=prb_units,
                    parent_mapping=parent_mapping,
                )
            )

            torsion_rows.append(
                _hostchar_build_torsion_row(
                    entry_name=entry_name,
                    m=prb_full,
                    ordered_units=ordered_probe_units,
                    expected_units=expected_units,
                )
            )

            print(
                f"Host characterization written for "
                f"{entry_name} "
                f"(host mapping RMSD {map_rmsd:.4f} A)"
            )

        except Exception as e:
            torsion_rows.append(
                _hostchar_empty_torsion_row(
                    entry_name,
                    expected_units,
                )
            )

            print(
                f"Host characterization skipped for "
                f"{entry_name}: {e}"
            )

    torsion_csv_path = os.path.join(
        host_char_dir,
        "host_torsion_angles.csv",
    )

    fieldnames = ["entry_name"]

    for i in range(
        1,
        expected_units + 1,
    ):
        fieldnames.extend([
            f"pyranose_{i}_atoms",
            f"phi{i}",
            f"psi{i}",
        ])

    with open(
        torsion_csv_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(torsion_rows)

    print(
        f"Host torsion angle CSV written: "
        f"{torsion_csv_path}"
    )

    return {
        "host_characterization_dir": host_char_dir,
        "host_torsion_angles_csv": torsion_csv_path,
    }

try:
    host_characterization_outputs = (
        export_host_characterization(
            out_dir=out_dir,
            records=records,
        )
    )
except Exception as e:
    print(
        f"Host characterization could not be generated: {e}"
    )
#
## ====================================================================================================