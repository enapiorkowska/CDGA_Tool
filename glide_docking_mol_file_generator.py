import sys
import re
from schrodinger import structure

pv_file = sys.argv[1]


def clean_filename(x):
    x = str(x).strip()
    x = re.sub(r'[\\/*?:"<>|]', "_", x)
    x = re.sub(r"\s+", "_", x)
    return x or "entry"


entries = list(structure.StructureReader(pv_file))

if not entries:
    raise RuntimeError("empty file")

print(f"Number of structures in the file: {len(entries)}")

receptor = entries[0]

for i, st in enumerate(entries[1:], start=1):
    entry_id = st.property.get("s_m_entry_id")
    if not entry_id:
        entry_id = f"id_{i}"

    pose_num = st.property.get("i_i_glide_posenum")
    if pose_num is None:
        pose_num = i

    score = st.property.get("r_i_glide_gscore")
    if score is None:
        score = 0.0
    else:
        score = float(score)

    complex_st = receptor.copy()
    complex_st.extend(st)

    # first line in the .mol file:
    # ID  score  pose_xxx
    complex_st.title = f"{entry_id}  {score:.3f}  pose_{pose_num}"

    filename = f"{clean_filename(entry_id)}_pose_{pose_num}.mol"

    print("Saving:", filename)

    with structure.StructureWriter(filename) as writer:
        writer.append(complex_st)

print("Done")