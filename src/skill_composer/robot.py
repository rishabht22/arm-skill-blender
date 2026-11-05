"""Shared, licensed Panda kinematics and collision-geometry scene builder."""
import copy
import xml.etree.ElementTree as ET
import numpy as np
from .common import ROOT

def vector(value):
    return " ".join(str(float(v)) for v in value)

def panda_scene():
    source = ROOT / "assets/panda/panda.xml"
    original = ET.parse(source).getroot()
    root = ET.Element("mujoco", model="two_panda_skill_sandbox")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(source.parent / "assets"), autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast", iterations="80")
    ET.SubElement(root, "visual").append(ET.Element("global", offwidth="960", offheight="720"))
    root.append(copy.deepcopy(original.find("default")))
    root.append(copy.deepcopy(original.find("asset")))
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse=".8 .8 .8")
    ET.SubElement(world, "geom", name="table", type="plane", size="2 2 .1", rgba=".16 .19 .23 1",
                  contype="3", conaffinity="3")
    actuator = ET.SubElement(root, "actuator")
    contact = ET.SubElement(root, "contact")
    for a, y in enumerate((-.75, .75)):
        body = copy.deepcopy(original.find("worldbody/body"))
        body.set("pos", f"0 {y} 0")
        yaw = np.pi / 2 if a == 0 else -np.pi / 2
        body.set("quat", vector([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        for element in body.iter():
            if "name" in element.attrib:
                element.set("name", f"a{a}_" + element.get("name"))
            if element.tag == "body":
                element.set("gravcomp", "1")
        for geom in body.iter("geom"):
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            geom.set("group", "1")
            geom.set("rgba", ".85 .88 .92 1" if a == 0 else ".28 .57 .78 1")
        hand = next(b for b in body.iter("body") if b.get("name") == f"a{a}_hand")
        ET.SubElement(hand, "site", name=f"a{a}_tcp", pos="0 0 .1034", size=".008", rgba="1 .6 .1 1")
        world.append(body)
        for joint in body.iter("joint"):
            name = joint.get("name")
            is_finger = "finger" in name
            ET.SubElement(actuator, "position", name=name + "_motor", joint=name,
                          kp="200" if is_finger else "900", kv="8" if is_finger else "70",
                          forcerange="-100 100")
        ET.SubElement(contact, "exclude", body1=f"a{a}_link0", body2=f"a{a}_link1")
    return root

