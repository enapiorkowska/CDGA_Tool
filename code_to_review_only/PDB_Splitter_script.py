## ====================================================================================================
# PDB FRAME SPLITTER 1.0.0
# PART OF PDB CONVERTER WORKFLOW
## ======================================================================
# This module splits multiframe PDB files into individual PDB files
# for further processing within the CDGA workflow.
#
#The Python scripts was developed with the assistance of ChatGPT 5.6 Sol, OpenAI). 
# All AI-generated code was reviewed, tested, and verified by the authors for accuracy, reliability, 
# and reproducibility.
#
# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# It uses third party libraries through their public APIs,
# including pandas and the KNIME Python Scripting API.
# These dependencies are distributed under their respective licenses.
#
# Input:
#
#   A KNIME table containing file paths in the "Location" column.
#
# Processing:
#
#   The PDB file is divided into separate frames using END and ENDMDL
#   records.
#
#   Only blocks containing ATOM or HETATM records are treated as
#   molecular frames.
#
#   CONECT records are preserved. If they are stored outside the
#   individual frames, they are copied to each generated PDB file.
#
#   MODEL, NUMMDL, END, and ENDMDL records are removed from the
#   generated frame files.
#
# Output:
#
#   Each frame is written as an individual PDB file using sequential
#   numbering:
#
#       source_0001.pdb
#       source_0002.pdb
#       source_0003.pdb
#
#   The KNIME output table contains the source file, frame number,
#   generated PDB path, frame name, expected MOL output name, and
#   processing status.
#
#   Generated files are stored under:
#
#       CDGA_Tool_processed_molfiles/
#           PDB_CONVERTER/
#               frame_files/
#
# ======================================================================
#
import knime.scripting.io as knio
import pandas as pd

from pathlib import Path
from urllib.parse import urlparse, unquote

import tempfile
import os


INPUT_COLUMN = "Location"

TEMP_ROOT = (
    Path(tempfile.gettempdir())
    / "CDGA_Tool_processed_molfiles"
    / "PDB_CONVERTER"
    / "frame_files"
)


def pdb_record(line):
    return line[:6].strip().upper()


def contains_atoms(lines):
    return any(
        pdb_record(line) in {"ATOM", "HETATM"}
        for line in lines
    )


def location_to_path(value):
    text = str(value).strip()

    if text.lower().startswith("file:"):
        parsed = urlparse(text)
        path_text = unquote(parsed.path)

        if os.name == "nt":
            if (
                len(path_text) >= 3
                and path_text[0] == "/"
                and path_text[2] == ":"
            ):
                path_text = path_text[1:]

            if parsed.netloc:
                path_text = (
                    "//"
                    + parsed.netloc
                    + path_text
                )

        return Path(path_text)

    return Path(text)


def split_pdb(lines):
    blocks = []
    current = []

    for line in lines:
        record = pdb_record(line)

        if record in {"END", "ENDMDL"}:
            if current:
                blocks.append(current)

            current = []

        else:
            current.append(line)

    if current:
        blocks.append(current)

    frames = []
    shared_conect = []

    for block in blocks:
        if contains_atoms(block):
            frames.append(block)

        else:
            for line in block:
                if pdb_record(line) == "CONECT":
                    shared_conect.append(line)

    return frames, shared_conect


input_df = knio.input_tables[0].to_pandas()

if INPUT_COLUMN not in input_df.columns:
    raise ValueError(
        f"Column '{INPUT_COLUMN}' was not found. "
        f"Available columns: {list(input_df.columns)}"
    )


TEMP_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

results = []


for _, row in input_df.iterrows():

    location = row[INPUT_COLUMN]

    if pd.isna(location):
        results.append(
            {
                "source_file": "",
                "frame": 0,
                "pdb_path": "",
                "frame_name": "",
                "output_name": "",
                "status": "Missing input path"
            }
        )
        continue


    try:
        source = location_to_path(location)

    except Exception as error:
        results.append(
            {
                "source_file": str(location),
                "frame": 0,
                "pdb_path": "",
                "frame_name": "",
                "output_name": "",
                "status": f"Invalid path: {error}"
            }
        )
        continue


    if not source.exists():
        results.append(
            {
                "source_file": str(source),
                "frame": 0,
                "pdb_path": "",
                "frame_name": "",
                "output_name": "",
                "status": "File not found"
            }
        )
        continue


    if not source.is_file():
        results.append(
            {
                "source_file": str(source),
                "frame": 0,
                "pdb_path": "",
                "frame_name": "",
                "output_name": "",
                "status": "Path is not a file"
            }
        )
        continue


    try:
        text = source.read_text(
            encoding="utf8",
            errors="replace"
        )

    except Exception as error:
        results.append(
            {
                "source_file": str(source),
                "frame": 0,
                "pdb_path": "",
                "frame_name": "",
                "output_name": "",
                "status": f"Could not read file: {error}"
            }
        )
        continue


    frames, shared_conect = split_pdb(
        text.splitlines()
    )


    if not frames:
        results.append(
            {
                "source_file": str(source),
                "frame": 0,
                "pdb_path": "",
                "frame_name": "",
                "output_name": "",
                "status": "No ATOM or HETATM records found"
            }
        )
        continue


    output_directory = (
        TEMP_ROOT
        / source.stem
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    for old_file in output_directory.glob(
        f"{source.stem}_*.pdb"
    ):
        try:
            old_file.unlink()
        except OSError:
            pass


    for frame_number, frame in enumerate(
        frames,
        start=1
    ):

        frame_name = (
            f"{source.stem}_{frame_number:04d}"
        )

        output_file = (
            output_directory
            / f"{frame_name}.pdb"
        )

        output_lines = [
            f"REMARK   FRAME: {frame_name}"
        ]

        local_conect = False


        for line in frame:
            record = pdb_record(line)

            if record in {
                "MODEL",
                "NUMMDL",
                "END",
                "ENDMDL"
            }:
                continue

            if record == "CONECT":
                local_conect = True

            output_lines.append(line)


        if not local_conect and shared_conect:
            output_lines.extend(shared_conect)


        output_lines.append("END")


        try:
            output_file.write_text(
                "\n".join(output_lines) + "\n",
                encoding="utf8"
            )

            status = "OK"

        except Exception as error:
            status = f"Could not write frame: {error}"


        results.append(
            {
                "source_file": str(source),
                "frame": frame_number,
                "pdb_path": str(output_file),
                "frame_name": frame_name,
                "output_name": f"{frame_name}.mol",
                "status": status
            }
        )


result_df = pd.DataFrame(
    results,
    columns=[
        "source_file",
        "frame",
        "pdb_path",
        "frame_name",
        "output_name",
        "status"
    ]
)


knio.output_tables[0] = knio.Table.from_pandas(
    result_df
)
#
## ====================================================================================================