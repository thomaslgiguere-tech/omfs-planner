"""Printer and resin profiles of the Impression workspace: data (JSON files), never constants in code.

Shipped profiles are in print_profiles/ next to this file; the user's own (new resins, edited values) are in
Blender's user config folder and win over a shipped profile of the same file name."""

import json
import re
from pathlib import Path

import bpy

SHIPPED = Path(__file__).parent / "print_profiles"


def user_folder(kind):
    base = bpy.utils.user_resource('CONFIG', path="omfs_planner/print_profiles", create=True)
    folder = Path(base) / kind
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def profiles(kind):
    """{file stem: profile dict} of 'printers' or 'resins', the user's replacing the shipped ones."""
    found = {}
    for folder in (SHIPPED / kind, user_folder(kind)):
        for path in sorted(folder.glob("*.json")):
            try:
                found[path.stem] = dict(json.loads(path.read_text()), user=folder != SHIPPED / kind)
            except (ValueError, OSError):
                pass   # an unreadable file is ignored, the others still load
    return found


def favourite_resin():
    resins = profiles("resins")
    return next((key for key, r in resins.items() if r.get("favourite")), next(iter(resins), ""))


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "resine"


def save_resin(resin, favourite=None):
    """Write a resin to the user's folder (its name gives the file name). favourite=True also unmarks the others."""
    resin = {k: v for k, v in resin.items() if k != "user"}
    if favourite:
        for key, other in profiles("resins").items():
            if other.get("favourite") and other["name"] != resin["name"]:
                other = {k: v for k, v in other.items() if k != "user"}
                (user_folder("resins") / f"{key}.json").write_text(json.dumps(dict(other, favourite=False), indent=1, ensure_ascii=False))
        resin["favourite"] = True
    key = slug(resin["name"])
    (user_folder("resins") / f"{key}.json").write_text(json.dumps(resin, indent=1, ensure_ascii=False))
    return key
