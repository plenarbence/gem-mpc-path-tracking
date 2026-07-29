#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import xacro


RENDERING_SENSOR_TYPES = {"camera", "depth", "multicamera"}


def generate_urdf(
    xacro_path: Path,
    robot_name: str,
    velodyne_points: bool,
    laser_points: bool,
) -> str:
    document = xacro.process_file(
        str(xacro_path),
        mappings={
            "robotname": robot_name,
            "velodyne_points": str(velodyne_points).lower(),
            "laser_points": str(laser_points).lower(),
        },
    )
    root = ET.fromstring(document.toxml())
    for gazebo_element in list(root.findall("gazebo")):
        sensor_types = {
            sensor.get("type", "") for sensor in gazebo_element.findall("sensor")
        }
        if sensor_types & RENDERING_SENSOR_TYPES:
            root.remove(gazebo_element)
    return ET.tostring(root, encoding="unicode")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a GEM URDF without image-rendering sensors."
    )
    parser.add_argument("--xacro", required=True, type=Path)
    parser.add_argument("--robot-name", default="gem")
    parser.add_argument("--velodyne-points", action="store_true")
    parser.add_argument("--laser-points", action="store_true")
    args = parser.parse_args()
    sys.stdout.write(
        generate_urdf(
            args.xacro,
            args.robot_name,
            args.velodyne_points,
            args.laser_points,
        )
    )


if __name__ == "__main__":
    main()
