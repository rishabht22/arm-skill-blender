"""MJCF for the laboratory workcell and the repository's Menagerie Panda."""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


PANDA = Path(__file__).resolve().parents[2] / "robofactory_lift/assets/panda"
HOME = np.array([0., -.56, 0., -2.35, 0., 1.79, np.pi / 4])
CARRIERS = 4
FLOOR = .127
STEM = .029
WORK = "process"
SCAN = "reader"
LAYOUTS = 128
FAMILIAR_LAYOUTS = 24


@dataclass(frozen=True)
class Layout:
    slots: dict
    process: tuple


def layout(index):
    if not 0 <= index < LAYOUTS:
        raise ValueError(index)
    rng = np.random.default_rng(52071 + index)
    dx, dy = rng.uniform(-.009, .009, (2, 7))
    process = (.53 + dx[0], .019 + dy[0])
    # The transfer split shifts the reader and both output racks outside
    # the position envelope used during fitting.
    relocated = index >= FAMILIAR_LAYOUTS
    slots = {
        WORK: (*process, FLOOR, 0.),
        SCAN: (.43 + dx[1] + .021 * relocated, .148 + dy[1] + .040 * relocated, FLOOR, 0.),
    }
    for i in range(CARRIERS):
        slots[f"feed{i}"] = (.35 + dx[2 + i % 2], -.270 + .102 * i + dy[2 + i % 2], FLOOR, 0.)
        slots[f"recovery{i}"] = (.281 + dx[2 + i % 2], -.270 + .102 * i + dy[2 + i % 2], FLOOR + .063, np.pi / 2)
        slots[f"release{i}"] = (.335 + dx[4], .119 + .075 * i + dy[4] + .019 * relocated, FLOOR, 0.)
        slots[f"reject{i}"] = (.459 + dx[5] + .019 * relocated, -.319 + .071 * i + dy[5], FLOOR, 0.)
    return Layout(slots, process)


def vec(values):
    return " ".join(f"{v:.8g}" for v in values)


def node(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {
        key: val if isinstance(val, str) else vec(val) if isinstance(val, (tuple, list, np.ndarray))
        else str(val) for key, val in attrs.items()
    })


def box(parent, name, pos, size, color, **kw):
    return node(parent, "geom", name=name, type="box", pos=pos, size=size, rgba=color, **kw)


FRAME = ".19 .23 .26 1"
STEEL = ".46 .56 .62 1"
TEAL = ".16 .54 .56 1"
AMBER = ".94 .65 .19 1"
RED = ".65 .24 .25 1"
DARK = ".08 .12 .15 1"
COLORS = (".24 .66 .86 1", ".86 .46 .24 1", ".44 .74 .41 1", ".67 .47 .85 1")


def socket(parent, label, color, *, scale=1.):
    """Polarized vertical bore. The +X wall has a dedicated rib pocket."""
    box(parent, f"{label}_floor", (0, 0, -.004), (.028, .027, .004), color)
    rails = [
        ("west", (-.021, 0, .008), (.006, .020, .008)),
        ("south", (0, -.017, .008), (.015, .005, .008)),
        ("north", (0, .017, .008), (.015, .005, .008)),
        ("east_a", (.021, -.0125, .008), (.006, .0075, .008)),
        ("east_b", (.021, .0125, .008), (.006, .0075, .008)),
        ("key", (.028, 0, .008), (.004, .005, .008)),
    ]
    for suffix, p, s in rails:
        box(parent, f"{label}_{suffix}", np.array(p) * [scale, scale, 1],
            np.array(s) * [scale, scale, 1], STEEL)


def slide(parent, name, axis, travel):
    node(parent, "joint", name=name, type="slide", axis=axis, range=(0, travel),
         damping=3, armature=.012, frictionloss=.12, solreflimit=".008 1")


def build(configuration):
    root = ET.parse(PANDA / "panda.xml").getroot()
    root.set("model", "Aster sample preparation workcell")
    root.find("compiler").set("meshdir", str(PANDA / "assets"))
    root.find("option").attrib.update(timestep=".004", iterations="18", cone="pyramidal")
    node(root, "visual")
    visual = root.find("visual")
    node(visual, "global", offwidth=1600, offheight=1000, azimuth=132, elevation=-34)
    node(visual, "quality", shadowsize=2048)
    world = root.find("worldbody")
    # A Cartesian site midway between the jaw pads.
    node(world.find(".//body[@name='hand']"), "site", name="tcp", pos=(0, 0, .104), size=.002)
    for body in world.findall(".//body"):
        body.set("gravcomp", "1")
    for side in ("left", "right"):
        finger = world.find(f".//body[@name='{side}_finger']")
        for geom in finger.findall("geom"):
            finger.remove(geom)
        box(finger, f"{side}_jaw", (0, .013, .019), (.009, .006, .021), STEEL)
        box(finger, f"{side}_pad", (0, .005, .045), (.0085, .0035, .009), DARK,
            friction="1.4 .015 .002")
    node(world, "light", pos=(.35, -.45, 1.5), dir=(0, .25, -1), diffuse=".85 .85 .85")
    node(world, "light", pos=(.75, .6, 1.1), dir=(-.4, -.5, -1), diffuse=".5 .55 .65")
    node(world, "camera", name="overview", pos=(1.25, -1.02, 1.0),
         xyaxes=".78 .62 0 -.29 .37 .88")
    box(world, "plinth", (.456, .01, .091), (.229, .425, .030), FRAME)
    box(world, "robot_plinth", (.03, 0, -.028), (.16, .14, .028), DARK)
    for tag, (x, y, z, _) in configuration.slots.items():
        if tag == WORK:
            continue
        fixture = node(world, "body", name=tag, pos=(x, y, z))
        if tag.startswith("recovery"):
            box(fixture, f"{tag}_riser", (0, 0, -.034), (.026, .022, .029), FRAME)
            box(fixture, f"{tag}_floor", (0, 0, -.004), (.032, .027, .004), AMBER)
            for side in (-1, 1):
                box(fixture, f"{tag}_rail{side}", (0, side * .022, .005), (.029, .004, .005), STEEL)
            continue
        socket(fixture, tag, TEAL if tag.startswith("release") else RED if tag.startswith("reject") else DARK)
        if tag == SCAN:
            # Break-beam sensing volume across the keyed read head.
            node(fixture, "site", name="read_beam", type="box", pos=(0, 0, .012),
                 size=(.023, .001, .001), rgba=".3 1 .6 .65")
            for side in (-1, 1):
                box(fixture, f"optics{side}", (.023, side * .029, .013),
                    (.009, .005, .012), DARK)
    x, y = configuration.process
    fixture = node(world, "body", name="instrument", pos=(x, y, FLOOR))
    box(fixture, "pedestal", (0, 0, -.005), (.076, .075, .002), FRAME)
    rotor = node(fixture, "body", name="rotor", pos=(0, 0, .006))
    node(rotor, "joint", name="rotor", axis=(0, 0, 1), range=(-.02, 1.59),
         damping=.13, frictionloss=.02, armature=.001)
    node(rotor, "geom", name="rotor_disc", type="cylinder", size=(.048, .004),
         pos=(0, 0, -.0042), rgba=STEEL)
    socket(rotor, WORK, DARK)
    box(rotor, "index_mark", (.039, 0, -.0001), (.005, .003, .001), AMBER, contype=0, conaffinity=0)
    for side in (-1, 1):
        if side > 0:
            box(fixture, "gantry_north", (.028, .057, .039), (.097, .005, .036), FRAME)
        else:
            # Machined relief for the moving latch tongue.
            box(fixture, "gantry_south_front", (-.014, -.057, .027), (.056, .005, .024), FRAME)
            box(fixture, "gantry_south_rear", (.098, -.057, .027), (.028, .005, .024), FRAME)
        box(fixture, f"guide{side}", (.036, side * .054, .076),
            (.091, .004, .004), STEEL, contype=0, conaffinity=0)
    lid = node(fixture, "body", name="lid", pos=(0, 0, .067))
    slide(lid, "lid", (1, 0, 0), .132)
    box(lid, "shield", (0, 0, 0), (.044, .050, .004), ".18 .58 .66 .65", mass=.055)
    box(lid, "lid_handle", (-.026, .033, .026), (.012, .010, .012), TEAL, mass=.016)
    box(lid, "lid_handle_post", (-.026, .033, .012), (.006, .006, .012), FRAME, mass=.01)
    # The bolt enters the path of this strike when the shield is closed.
    box(lid, "strike", (.031, -.052, -.008), (.008, .009, .005), STEEL, mass=.014, solref=".008 1", solimp=".99 .9999 .0001")
    bolt = node(fixture, "body", name="bolt", pos=(.054, -.083, .059))
    slide(bolt, "bolt", (0, 1, 0), .026)
    box(bolt, "bolt_pin", (0, 0, 0), (.007, .009, .005), AMBER, mass=.02, solref=".008 1", solimp=".99 .9999 .0001")
    box(bolt, "bolt_handle", (0, -.022, .039), (.011, .009, .010), AMBER, mass=.015)
    box(bolt, "bolt_stem", (0, -.013, .017), (.005, .005, .018), FRAME, mass=.012)
    # External spring-return plunger signals the electromechanical index drive.
    button = node(fixture, "body", name="button", pos=(-.049, .074, .067))
    node(button, "joint", name="button", type="slide", axis=(0, 0, -1), range=(0, .013),
         stiffness=460, damping=3, armature=.003)
    box(button, "button_cap", (0, 0, 0), (.010, .010, .005), RED, mass=.013)
    box(fixture, "panel", (-.049, .074, .045), (.019, .019, .008), FRAME)
    equalities = node(root, "equality")
    for handle in ("lid", "bolt"):
        node(equalities, "weld", name=f"grip_{handle}", body1="hand", body2=handle,
             active="false", solref=".009 1", torquescale=".32", solimp=".98 .999 .0002")
    for i, color in enumerate(COLORS):
        x, y, z, _ = configuration.slots[f"feed{i}"]
        label = f"c{i}"
        carrier = node(world, "body", name=label, pos=(x, y, z + STEM + .0004))
        node(carrier, "freejoint", name=label)
        box(carrier, f"{label}_grip", (0, 0, .011), (.011, .010, .012), color,
            mass=.045, friction="1.8 .02 .002")
        box(carrier, f"{label}_flange", (0, 0, -.005), (.021, .017, .004), color, mass=.018)
        box(carrier, f"{label}_stem", (0, 0, -.019), (.012, .009, .010), DARK, mass=.016, group=2)
        box(carrier, f"{label}_rib", (.0155, 0, -.020), (.0045, .003, .008), AMBER, mass=.005)
        node(equalities, "weld", name=f"grip_{label}", body1="hand", body2=label,
             active="false", solref=".009 1", torquescale=".32", solimp=".98 .999 .0002")
    actuators = node(root, "actuator")
    for i, force in enumerate((87, 87, 87, 87, 26, 22, 16), 1):
        node(actuators, "position", name=f"arm{i}", joint=f"joint{i}",
             kp=1550 if i < 5 else 750, kv=49 if i < 5 else 23,
             forcerange=(-force, force))
    for i in (1, 2):
        node(actuators, "position", name=f"jaw{i}", joint=f"finger_joint{i}",
             kp=600, kv=9, ctrlrange=(0, .038), forcerange=(-9, 14))
    node(actuators, "position", name="index", joint="rotor", kp=8, kv=.5, forcerange=(-1, 1))
    # Arm links retain collision against fixtures. Filter the plinth/base overlap.
    contact = node(root, "contact")
    node(contact, "exclude", body1="link0", body2="world")
    return ET.tostring(root, encoding="unicode")
