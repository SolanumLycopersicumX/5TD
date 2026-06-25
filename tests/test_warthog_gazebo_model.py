import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SDF_PATH = ROOT / "sim/gazebo/models/warthog_02m_pro_4wd_chassis/model.sdf"
URDF_PATH = ROOT / "sim/urdf/warthog_02m_pro_4wd_chassis.urdf"
WORLD_PATH = ROOT / "sim/gazebo/worlds/warthog_flat_test.sdf"
LAUNCH_SCRIPT = ROOT / "sim/gazebo/run_warthog_flat_test.sh"

EXPECTED_WHEELS = {
    "front_left": (0.343, 0.500, 0.285),
    "front_right": (0.343, -0.500, 0.285),
    "rear_left": (-0.498, 0.500, 0.285),
    "rear_right": (-0.498, -0.500, 0.285),
}


class WarthogGazeboModelTest(unittest.TestCase):
    def test_sdf_wheel_links_match_cad_mount_positions(self):
        root = ET.parse(SDF_PATH).getroot()
        for wheel, expected in EXPECTED_WHEELS.items():
            link = root.find(f".//link[@name='{wheel}_wheel_link']")
            self.assertIsNotNone(link, wheel)
            pose = [float(value) for value in link.findtext("pose").split()[:3]]
            self.assertEqual([round(value, 3) for value in pose], list(expected))

            radius = float(link.findtext(".//cylinder/radius"))
            length = float(link.findtext(".//cylinder/length"))
            self.assertAlmostEqual(radius, 0.285, places=3)
            self.assertAlmostEqual(length, 0.220, places=3)

    def test_sdf_diff_drive_uses_same_wheel_geometry(self):
        root = ET.parse(SDF_PATH).getroot()
        plugin = root.find(".//plugin[@name='gz::sim::systems::DiffDrive']")
        self.assertIsNotNone(plugin)
        self.assertAlmostEqual(float(plugin.findtext("wheel_radius")), 0.285, places=3)
        self.assertAlmostEqual(float(plugin.findtext("wheel_separation")), 1.000, places=3)
        self.assertAlmostEqual(float(plugin.findtext("max_linear_velocity")), 0.30, places=3)
        self.assertAlmostEqual(float(plugin.findtext("max_linear_acceleration")), 0.50, places=3)
        self.assertAlmostEqual(float(plugin.findtext("max_angular_velocity")), 0.80, places=3)
        self.assertAlmostEqual(float(plugin.findtext("max_angular_acceleration")), 1.00, places=3)

    def test_wheel_collisions_define_rolling_friction_direction(self):
        root = ET.parse(SDF_PATH).getroot()
        for wheel in EXPECTED_WHEELS:
            collision = root.find(f".//link[@name='{wheel}_wheel_link']/collision")
            self.assertIsNotNone(collision, wheel)
            fdir1 = collision.findtext(".//friction/ode/fdir1")
            self.assertEqual(fdir1, "0 0 1", wheel)

    def test_sdf_wheel_axes_follow_diff_drive_convention(self):
        root = ET.parse(SDF_PATH).getroot()
        for wheel in EXPECTED_WHEELS:
            link = root.find(f".//link[@name='{wheel}_wheel_link']")
            self.assertIsNotNone(link, wheel)
            rpy = [float(value) for value in link.findtext("pose").split()[3:6]]
            self.assertEqual([round(value, 6) for value in rpy], [-1.570796, 0.0, 0.0], wheel)

            joint = root.find(f".//joint[@name='{wheel}_wheel_joint']")
            self.assertIsNotNone(joint, wheel)
            axis = joint.findtext("axis/xyz")
            self.assertEqual(axis, "0 0 1", wheel)

    def test_urdf_joint_positions_match_sdf_wheels(self):
        root = ET.parse(URDF_PATH).getroot()
        for wheel, expected in EXPECTED_WHEELS.items():
            joint = root.find(f".//joint[@name='{wheel}_wheel_joint']")
            self.assertIsNotNone(joint, wheel)
            origin = joint.find("origin")
            xyz = [float(value) for value in origin.attrib["xyz"].split()]
            self.assertEqual([round(value, 3) for value in xyz], list(expected))

    def test_wheel_links_use_cad_visual_only(self):
        sdf_root = ET.parse(SDF_PATH).getroot()
        for wheel in EXPECTED_WHEELS:
            link = sdf_root.find(f".//link[@name='{wheel}_wheel_link']")
            self.assertIsNotNone(link.find("collision"), wheel)
            self.assertIsNone(link.find("visual"), wheel)

        urdf_root = ET.parse(URDF_PATH).getroot()
        for wheel in EXPECTED_WHEELS:
            link = urdf_root.find(f".//link[@name='{wheel}_wheel_link']")
            self.assertIsNotNone(link.find("collision"), wheel)
            self.assertIsNone(link.find("visual"), wheel)

    def test_flat_test_world_has_ground_and_model(self):
        root = ET.parse(WORLD_PATH).getroot()
        world = root.find("world")
        self.assertIsNotNone(world)
        self.assertEqual(world.attrib["name"], "warthog_flat_test")

        ground = world.find(".//model[@name='ground_plane']")
        self.assertIsNotNone(ground)
        self.assertIsNotNone(ground.find(".//collision/geometry/plane"))
        self.assertIsNotNone(ground.find(".//visual/geometry/plane"))

        include_uris = [uri.text for uri in world.findall(".//include/uri")]
        self.assertIn("model://warthog_02m_pro_4wd_chassis", include_uris)

    def test_launch_script_sources_ros_without_nounset(self):
        lines = LAUNCH_SCRIPT.read_text().splitlines()
        source_index = lines.index("source /opt/ros/jazzy/setup.bash")
        self.assertIn("set +u", lines[:source_index])
        self.assertIn("set -u", lines[source_index + 1 :])


if __name__ == "__main__":
    unittest.main()
