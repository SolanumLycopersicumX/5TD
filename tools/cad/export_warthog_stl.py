#!/usr/bin/env python3
"""Export the Warthog STEP chassis assembly to a centered STL visual mesh.

Run with FreeCAD's Python interpreter, for example:

    freecad.cmd tools/cad/export_warthog_stl.py \
        --input "assets/cad/warthog_02m_pro_4wd_chassis/Warthog-02M-Pro 4WD Chassis.STEP" \
        --output sim/gazebo/models/warthog_02m_pro_4wd_chassis/meshes/chassis_visual.stl

The exported STL keeps millimeter units. URDF/SDF files should use scale
`0.001 0.001 0.001` when loading it in meter-based simulators.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import FreeCAD as App
import MeshPart
import Part


def _bbox_dict(bound_box):
    return {
        "min_mm": [bound_box.XMin, bound_box.YMin, bound_box.ZMin],
        "max_mm": [bound_box.XMax, bound_box.YMax, bound_box.ZMax],
        "span_mm": [bound_box.XLength, bound_box.YLength, bound_box.ZLength],
    }


def _center_shape_on_ground(shape, rotate_y_to_x):
    bbox = shape.BoundBox
    centered = shape.copy()
    centered.translate(
        App.Vector(
            -(bbox.XMin + bbox.XMax) / 2.0,
            -(bbox.YMin + bbox.YMax) / 2.0,
            -bbox.ZMin,
        )
    )
    if rotate_y_to_x:
        centered.rotate(App.Vector(0, 0, 0), App.Vector(0, 0, 1), -90.0)
    return centered


def export_stl(args):
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    metadata_path = Path(args.metadata).resolve() if args.metadata else output_path.with_suffix(".metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    shape = Part.Shape()
    shape.read(str(input_path))
    normalized = _center_shape_on_ground(shape, rotate_y_to_x=args.rotate_y_to_x)

    mesh = MeshPart.meshFromShape(
        Shape=normalized,
        LinearDeflection=args.linear_deflection_mm,
        AngularDeflection=args.angular_deflection_rad,
        Relative=False,
    )
    mesh.write(str(output_path))

    metadata = {
        "source_step": str(input_path),
        "output_stl": str(output_path),
        "stl_units": "millimeter",
        "gazebo_mesh_scale": [0.001, 0.001, 0.001],
        "rotate_original_y_axis_to_gazebo_x_axis": args.rotate_y_to_x,
        "linear_deflection_mm": args.linear_deflection_mm,
        "angular_deflection_rad": args.angular_deflection_rad,
        "source_solid_count": len(shape.Solids),
        "source_bbox": _bbox_dict(shape.BoundBox),
        "normalized_bbox": _bbox_dict(normalized.BoundBox),
        "mesh_facets": mesh.CountFacets,
        "mesh_points": mesh.CountPoints,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source STEP/IGES path")
    parser.add_argument("--output", required=True, help="Output STL path")
    parser.add_argument("--metadata", help="Optional output metadata JSON path")
    parser.add_argument("--linear-deflection-mm", type=float, default=8.0)
    parser.add_argument("--angular-deflection-rad", type=float, default=0.35)
    parser.add_argument(
        "--no-rotate-y-to-x",
        dest="rotate_y_to_x",
        action="store_false",
        help="Keep the CAD assembly axes instead of rotating CAD Y into Gazebo X",
    )
    parser.set_defaults(rotate_y_to_x=True)
    return parser.parse_args()


if __name__ == "__main__":
    export_stl(parse_args())
