import numpy as np

from app.config import apply_environment, settings

# Anny reads its cache folder when it is imported.
apply_environment(settings)

import anny
import trimesh


# Chest plane offsets above the waist, as a fraction of body height.
CHEST_TARGET_OFFSET = 0.145
CHEST_OFFSETS = np.arange(0.135, 0.151, 0.001)

# Hip plane offset below the waist, as a fraction of body height.
HIP_OFFSET = 0.10

# Shoulder tip search box around each shoulder joint (metres), see shoulder_points.
SHOULDER_POINT_HALF_WIDTH = 0.015
SHOULDER_POINT_HALF_DEPTH = 0.03

# Tape-style chest and hip (male fit only, see BodyMeasurements.tape_measurements).
# Nipple vertices of Anny's topology (mirror pair, most anterior chest points of
# an adult male; fixed indices like Anny's own waist vertices).
NIPPLE_VERTICES = (8526, 1870)
# Bones whose vertices are arm geometry (excluded from torso girths).
ARM_BONE_PREFIXES = ("upperarm", "lowerarm", "wrist", "finger", "metacarpal", "thumb")
# Hip: the tape girth at the level of greatest buttock prominence (ISO 8559-1
# hip girth): the most posterior skin point between HIP_BAND_BELOW_WAIST and
# HIP_BAND_DEPTH below the waist, HIP_BAND_X_MIN..HIP_BAND_X_MAX from the
# midline. The largest girth is not used: in Anny's rest pose the legs stand
# apart and, just above the crotch, the thigh roots read up to 4 cm more.
HIP_BAND_BELOW_WAIST = 0.05
HIP_BAND_DEPTH = 0.30
HIP_BAND_X_MIN = 0.02
HIP_BAND_X_MAX = 0.15
# If that level is already split into the legs, step up by this much.
HIP_STEP_UP = 0.005
# A slice loop belongs to one leg when its centre is this far from the midline.
LEG_LOOP_OFFSET = 0.03

# The upper-arm loop's centre must lie within this distance of the arm bone.
UPPER_ARM_MAX_OFFSET = 0.06


def _polygon_contains_origin(polygon):
    """Ray-casting test: is (0, 0) inside the closed 2-D polygon?"""

    x, y = polygon[:, 0], polygon[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)

    crosses = (y > 0) != (y2 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at_zero = x + (0 - y) * (x2 - x) / (y2 - y)

    return bool(np.count_nonzero(crosses & (x_at_zero > 0)) % 2 == 1)


class BodyMeasurements:
    def __init__(self, model):
        self.model = model
        self.anthropometry = anny.Anthropometry(model)

        self._faces = (
            model.faces
            .detach()
            .cpu()
            .numpy()
        )

        self._bone_index = {
            label: index
            for index, label in enumerate(model.bone_labels)
        }

        # Faces of the body without the arms, for tape girths: a vertex is
        # arm geometry when most of its skinning weight is on arm bones.
        arm_bones = [
            index
            for index, label in enumerate(model.bone_labels)
            if label.startswith(ARM_BONE_PREFIXES)
        ]
        weights = model.vertex_bone_weights.detach().cpu().numpy()
        indices = model.vertex_bone_indices.detach().cpu().numpy()
        arm_vertices = np.where(np.isin(indices, arm_bones), weights, 0.0).sum(axis=1) > 0.5
        self._body_faces = self._faces[~arm_vertices[self._faces].any(axis=1)]

    # ============================================================
    # SHARED CONTEXT
    # ============================================================

    def _context(self, rest_vertices):
        """
        Everything that several measurements need from the same body.

        Building the mesh and running the anthropometry once per body
        avoids repeating that work for every individual measurement.
        """

        base = self.anthropometry(rest_vertices)

        vertices = (
            rest_vertices[0]
            .detach()
            .cpu()
            .numpy()
        )

        waist_indices = (
            self.anthropometry.waist_vertex_indices
        )

        return {
            "base": base,
            "height": float(base["height"].item()),
            "vertices": vertices,
            "waist_z": float(vertices[waist_indices, 2].mean()),
            "mesh": self._mesh(rest_vertices),
        }

    def _mesh(self, rest_vertices):
        vertices = rest_vertices[0].detach().cpu().numpy()

        return trimesh.Trimesh(
            vertices=vertices,
            faces=self._faces,
            process=False,
        )

    def _perimeter(self, points):
        closed = np.vstack([points, points[0]])

        return float(
            np.linalg.norm(
                np.diff(closed, axis=0),
                axis=1,
            ).sum()
        )

    def _loop_area(self, points):
        return float(
            0.5
            * abs(
                np.dot(
                    points[:, 0],
                    np.roll(points[:, 1], -1),
                )
                - np.dot(
                    points[:, 1],
                    np.roll(points[:, 0], -1),
                )
            )
        )

    def _section(self, rest_vertices, origin, normal, mesh=None):
        if mesh is None:
            mesh = self._mesh(rest_vertices)

        origin = np.asarray(origin, dtype=float)
        normal = np.asarray(normal, dtype=float)

        length = np.linalg.norm(normal)

        if length < 1e-8:
            return None

        normal = normal / length

        section = mesh.section(
            plane_origin=origin,
            plane_normal=normal,
        )

        if section is None:
            return None

        candidates = []

        for points in section.discrete:
            if len(points) < 3:
                continue

            perimeter = self._perimeter(points)
            area = self._loop_area(points)

            width = (
                points[:, 0].max()
                - points[:, 0].min()
            )

            candidates.append(
                {
                    "area": area,
                    "circumference": perimeter,
                    "width": float(width),
                    "points": points,
                }
            )

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda x: x["area"],
        )

        return {
            "circumference": best["circumference"],
            "width": best["width"],
            "points": best["points"],
        }

    def _horizontal_section(self, rest_vertices, z, mesh=None):
        return self._section(
            rest_vertices,
            [0.0, 0.0, z],
            [0.0, 0.0, 1.0],
            mesh=mesh,
        )

    def _waist_z(self, rest_vertices):
        vertices = (
            rest_vertices[0]
            .detach()
            .cpu()
            .numpy()
        )

        indices = self.anthropometry.waist_vertex_indices

        return float(
            vertices[indices, 2].mean()
        )

    def _chest_loop(self, mesh, z):
        """Largest valid torso loop at height z, or None."""

        section = mesh.section(
            plane_origin=[0.0, 0.0, z],
            plane_normal=[0.0, 0.0, 1.0],
        )

        if section is None:
            return None

        loops = []

        for points in section.discrete:

            if len(points) < 6:
                continue

            if abs(
                float(points[:, 0].mean())
            ) > 0.05:
                continue

            width = (
                points[:, 0].max()
                - points[:, 0].min()
            )

            # Reject oversized torso loops.
            if width > 0.45:
                continue

            loops.append(
                {
                    "z": float(z),
                    "circumference": self._perimeter(points),
                    "width": float(width),
                    "area": self._loop_area(points),
                }
            )

        if not loops:
            return None

        return max(
            loops,
            key=lambda x: x["area"],
        )

    def _chest_section(self, rest_vertices, context=None):
        """
        Stable chest measurement.

        The chest is measured around a narrow anatomical
        band above the waist instead of selecting the largest
        circumference from a wide scan.

        This prevents the selected chest region from jumping
        between different torso loops as body weight changes.

        The result is the valid loop closest to the target height
        (ties broken by the larger area). Planes are visited from the
        target outwards and the scan stops as soon as no remaining
        plane can be closer, so usually only one plane is sliced
        instead of all sixteen. The selected loop is the same one
        the full scan would choose.
        """

        if context is None:
            context = self._context(rest_vertices)

        height = context["height"]
        waist_z = context["waist_z"]
        mesh = context["mesh"]

        target_z = (
            waist_z
            + height * CHEST_TARGET_OFFSET
        )

        planes = sorted(
            (
                abs(z - target_z),
                z,
            )
            for z in (
                waist_z + height * offset
                for offset in CHEST_OFFSETS
            )
        )

        best = None

        for distance, z in planes:
            if best is not None and distance > best["distance"]:
                break

            loop = self._chest_loop(mesh, z)

            if loop is None:
                continue

            loop["distance"] = distance

            if (
                best is None
                or (distance, -loop["area"])
                < (best["distance"], -best["area"])
            ):
                best = loop

        if best is None:
            return None

        return {
            "z": best["z"],
            "circumference": best["circumference"],
            "width": best["width"],
            "points": None,
        }

    def _hip_section(self, rest_vertices, context=None):
        if context is None:
            context = self._context(rest_vertices)

        hip_z = (
            context["waist_z"]
            - context["height"] * HIP_OFFSET
        )

        section = context["mesh"].section(
            plane_origin=[0.0, 0.0, hip_z],
            plane_normal=[0.0, 0.0, 1.0],
        )

        if section is None:
            return None

        candidates = []

        for points in section.discrete:
            if len(points) < 20:
                continue

            if abs(points[:, 0].mean()) >= 0.1:
                continue

            perimeter = self._perimeter(points)

            width = (
                points[:, 0].max()
                - points[:, 0].min()
            )

            candidates.append(
                {
                    "circumference": perimeter,
                    "width": float(width),
                    "points": points,
                }
            )

        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda x: x["circumference"],
        )

        return {
            "z": float(hip_z),
            "circumference": best["circumference"],
            "width": best["width"],
        }

    def _bone_origin(self, rest_bone_poses, index):
        pose = (
            rest_bone_poses[0, index]
            .detach()
            .cpu()
            .numpy()
        )

        return pose[:3, 3]

    def _bone_axis(
        self,
        rest_bone_poses,
        start_index,
        end_index,
    ):
        start = self._bone_origin(
            rest_bone_poses,
            start_index,
        )

        end = self._bone_origin(
            rest_bone_poses,
            end_index,
        )

        axis = end - start

        length = np.linalg.norm(axis)

        if length < 1e-8:
            return None

        return axis / length

    def _limb_sections(
        self,
        rest_vertices,
        rest_bone_poses,
        start_index,
        end_index,
        t,
        mesh=None,
    ):
        start = self._bone_origin(
            rest_bone_poses,
            start_index,
        )

        end = self._bone_origin(
            rest_bone_poses,
            end_index,
        )

        origin = start + (end - start) * t

        axis = self._bone_axis(
            rest_bone_poses,
            start_index,
            end_index,
        )

        if axis is None:
            return []

        if mesh is None:
            mesh = self._mesh(rest_vertices)

        section = mesh.section(
            plane_origin=origin,
            plane_normal=axis,
        )

        if section is None:
            return []

        results = []

        for points in section.discrete:
            if len(points) < 10:
                continue

            perimeter = self._perimeter(points)

            results.append(
                {
                    "circumference": perimeter,
                    "points": points,
                }
            )

        return results

    def _smallest_limb_section(
        self,
        rest_vertices,
        rest_bone_poses,
        start_index,
        end_index,
        t,
        mesh=None,
    ):
        sections = self._limb_sections(
            rest_vertices,
            rest_bone_poses,
            start_index,
            end_index,
            t,
            mesh=mesh,
        )

        if not sections:
            return None

        return min(
            sections,
            key=lambda x: x["circumference"],
        )

    def _upper_arm_section(
        self,
        rest_vertices,
        rest_bone_poses,
        mesh=None,
        side=".L",
    ):
        """
        Upper-arm circumference at mid upper arm.

        The plane is perpendicular to the arm, halfway between the
        shoulder joint (upperarm01) and the elbow (lowerarm01). Only a
        loop around the arm bone counts: the bone axis must lie inside
        it and its centre within UPPER_ARM_MAX_OFFSET of the axis. A loop
        that has merged with the torso through the armpit fails these
        checks, and the measurement is reported as missing, not wrong.
        """

        start = self._bone_index.get("upperarm01" + side)
        end = self._bone_index.get("lowerarm01" + side)

        if start is None or end is None:
            return None

        sections = self._limb_sections(
            rest_vertices,
            rest_bone_poses,
            start,
            end,
            0.5,
            mesh=mesh,
        )

        if not sections:
            return None

        a = self._bone_origin(rest_bone_poses, start)
        b = self._bone_origin(rest_bone_poses, end)
        axis = (b - a) / np.linalg.norm(b - a)
        origin = a + (b - a) * 0.5

        u = np.cross(axis, [0.0, 0.0, 1.0])

        if np.linalg.norm(u) < 1e-6:
            u = np.cross(axis, [1.0, 0.0, 0.0])

        u /= np.linalg.norm(u)
        v = np.cross(axis, u)

        best = None
        best_offset = None

        for section in sections:
            points = section["points"] - origin
            centre = points.mean(axis=0)
            offset = np.linalg.norm(centre - (centre @ axis) * axis)

            if offset > UPPER_ARM_MAX_OFFSET:
                continue

            if not _polygon_contains_origin(np.c_[points @ u, points @ v]):
                continue

            if best is None or offset < best_offset:
                best = section
                best_offset = offset

        return best

    def _leg_section(
        self,
        rest_vertices,
        rest_bone_poses,
        start_label,
        end_label,
        side,
        t=0.5,
        mesh=None,
    ):
        """
        Circumference of one leg, measured on that leg only.

        The plane is perpendicular to this side's own bone, from
        start_label + side to end_label + side (looked up by name), at
        fraction t. Only a loop around that bone counts: the bone axis
        must lie inside it, and the loop whose centre is closest to the
        axis is kept. A plane from one leg would cut the other leg at an
        angle, so each side gets its own plane, and the side of the
        result is the side of the bones, not a guess from loop size.
        Returns None if the bones or a loop around them are missing.
        """

        start = self._bone_index.get(start_label + side)
        end = self._bone_index.get(end_label + side)

        if start is None or end is None:
            return None

        sections = self._limb_sections(
            rest_vertices,
            rest_bone_poses,
            start,
            end,
            t,
            mesh=mesh,
        )

        if not sections:
            return None

        a = self._bone_origin(rest_bone_poses, start)
        b = self._bone_origin(rest_bone_poses, end)
        axis = (b - a) / np.linalg.norm(b - a)
        origin = a + (b - a) * t

        u = np.cross(axis, [0.0, 0.0, 1.0])

        if np.linalg.norm(u) < 1e-6:
            u = np.cross(axis, [1.0, 0.0, 0.0])

        u /= np.linalg.norm(u)
        v = np.cross(axis, u)

        best = None
        best_offset = None

        for section in sections:
            points = section["points"] - origin

            if not _polygon_contains_origin(np.c_[points @ u, points @ v]):
                continue

            centre = points.mean(axis=0)
            offset = np.linalg.norm(centre - (centre @ axis) * axis)

            if best is None or offset < best_offset:
                best = section
                best_offset = offset

        return None if best is None else best["circumference"]

    def _neck_section(
        self,
        rest_vertices,
        rest_bone_poses,
        mesh=None,
    ):
        """
        Neck circumference halfway along the neck.

        The plane is perpendicular to the neck01 -> head bone axis at its
        midpoint, so it follows each body's own neck instead of a fixed
        height. Only a loop around that axis counts (same selection as
        _leg_section, which builds the bone names as label + side; the neck
        bones have no side). Returns None if the bones or the loop are
        missing.
        """

        return self._leg_section(
            rest_vertices,
            rest_bone_poses,
            "neck01",
            "head",
            "",
            t=0.5,
            mesh=mesh,
        )

    # ============================================================
    # JOINT-BASED LENGTHS (ESTIMATES)
    # ============================================================

    def _joint_distance(self, rest_bone_poses, start_label, end_label):
        start = self._bone_index.get(start_label)
        end = self._bone_index.get(end_label)

        if start is None or end is None:
            return None

        distance = np.linalg.norm(
            self._bone_origin(rest_bone_poses, end)
            - self._bone_origin(rest_bone_poses, start)
        )

        return float(distance)

    def _mean_of_sides(self, rest_bone_poses, start_label, end_label):
        values = [
            self._joint_distance(
                rest_bone_poses,
                start_label + side,
                end_label + side,
            )
            for side in (".L", ".R")
        ]

        values = [x for x in values if x is not None]

        if not values:
            return None

        return float(np.mean(values))

    def joint_lengths(self, rest_bone_poses):
        """
        Skeleton-based length estimates.

        These are distances between joint centres, not tape
        measurements on the skin:

        - shoulder width: left to right shoulder joint
          (upperarm01 heads), narrower than a tape-measured
          shoulder width over the acromion.
        - arm length: shoulder joint to wrist joint.
        - leg length: hip joint (upperleg01 head) to ankle
          joint (foot head).
        """

        return {
            "shoulder_width_m": self._joint_distance(
                rest_bone_poses,
                "upperarm01.L",
                "upperarm01.R",
            ),
            "arm_length_m": self._mean_of_sides(
                rest_bone_poses,
                "upperarm01",
                "wrist",
            ),
            "leg_length_m": self._mean_of_sides(
                rest_bone_poses,
                "upperleg01",
                "foot",
            ),
        }

    # ============================================================
    # SHOULDER BREADTH (SURFACE)
    # ============================================================

    def shoulder_points(self, vertices, rest_bone_poses):
        """
        Shoulder tips (acromion proxies), one per side: the highest skin
        point directly above the shoulder joint (upperarm01 head), within
        SHOULDER_POINT_HALF_WIDTH sideways and SHOULDER_POINT_HALF_DEPTH
        front-to-back of it. Rest pose, z up. None if a side has no
        such point.
        """

        points = []

        for side in (".L", ".R"):
            index = self._bone_index.get("upperarm01" + side)

            if index is None:
                return None

            joint = self._bone_origin(rest_bone_poses, index)

            near = (
                (np.abs(vertices[:, 0] - joint[0]) < SHOULDER_POINT_HALF_WIDTH)
                & (np.abs(vertices[:, 1] - joint[1]) < SHOULDER_POINT_HALF_DEPTH)
                & (vertices[:, 2] > joint[2])
            )

            if not np.any(near):
                return None

            candidates = vertices[near]
            points.append(candidates[np.argmax(candidates[:, 2])])

        return points

    def shoulder_breadth(self, vertices, rest_bone_poses):
        """
        Shoulder breadth on the skin: horizontal straight-line distance
        between the two shoulder tips, as measured with a ruler or
        calliper from shoulder tip to shoulder tip across the back.
        Wider than shoulder_width_m (joint centres).
        """

        points = self.shoulder_points(vertices, rest_bone_poses)

        if points is None:
            return None

        left, right = points

        return float(np.hypot(left[0] - right[0], left[1] - right[1]))

    # ============================================================
    # TAPE-STYLE GIRTHS (ADULT MALE FIT)
    # ============================================================

    @staticmethod
    def _hull_perimeter(points):
        """Length of a tape around the points: their convex outline in the horizontal plane."""

        if len(points) < 3:
            return None

        from scipy.spatial import ConvexHull

        outline = points[ConvexHull(points[:, :2]).vertices, :2]

        return float(
            np.linalg.norm(np.diff(np.vstack([outline, outline[:1]]), axis=0), axis=1).sum()
        )

    def _tape_at(self, body, z, stop_at_legs=False):
        """Tape girth of the body at height z; None at the crotch when stop_at_legs."""

        section = body.section(plane_origin=[0.0, 0.0, z], plane_normal=[0.0, 0.0, 1.0])

        if section is None:
            return None

        loops = [np.asarray(loop) for loop in section.discrete if len(loop) > 10]

        if not loops:
            return None

        if stop_at_legs and len(loops) > 1 and all(
            abs(loop[:, 0].mean()) > LEG_LOOP_OFFSET for loop in loops
        ):
            return None

        return self._hull_perimeter(np.vstack(loops))

    def tape_measurements(self, output):
        """
        Chest and hip as measured with a tape on a person standing with the
        arms down (used by the adult male fit; measure() is unchanged):

        - chest_tape_m: around the torso at nipple height (NIPPLE_VERTICES),
          the arms left out, the tape bridging the hollows (convex outline);
        - hip_tape_m: the same around the hips at the level of greatest
          buttock prominence (see HIP_BAND_*).

        measure()'s chest_m is sliced higher, at armpit level, where in the
        rest pose the torso joins the raised arms; it reads 6-13 cm more
        than a tape for adult men and fails on bulky torsos.
        """

        rest_vertices = output["rest_vertices"]
        vertices = rest_vertices[0].detach().cpu().numpy()
        body = trimesh.Trimesh(vertices=vertices, faces=self._body_faces, process=False)

        chest_z = float(vertices[list(NIPPLE_VERTICES), 2].mean())
        chest = self._tape_at(body, chest_z)

        waist_z = float(vertices[self.anthropometry.waist_vertex_indices, 2].mean())

        band = (
            (vertices[:, 2] < waist_z - HIP_BAND_BELOW_WAIST)
            & (vertices[:, 2] > waist_z - HIP_BAND_DEPTH)
            & (np.abs(vertices[:, 0]) > HIP_BAND_X_MIN)
            & (np.abs(vertices[:, 0]) < HIP_BAND_X_MAX)
        )

        hip = None
        hip_z = None

        if np.any(band):
            # Most posterior point (the back is +y).
            z = float(vertices[band][np.argmax(vertices[band][:, 1]), 2])

            while z < waist_z and hip is None:
                hip = self._tape_at(body, z, stop_at_legs=True)
                hip_z = z
                z += HIP_STEP_UP

        return {
            "chest_tape_m": chest,
            "chest_tape_z": chest_z,
            "hip_tape_m": hip,
            "hip_tape_z": hip_z,
        }

    def measure(self, output):
        rest_vertices = output["rest_vertices"]
        rest_bone_poses = output["rest_bone_poses"]

        context = self._context(rest_vertices)

        base = context["base"]
        mesh = context["mesh"]
        vertices = context["vertices"]

        height = context["height"]

        mass = float(
            base["mass"].item()
        )

        bmi = float(
            base["bmi"].item()
        )

        waist_indices = (
            self.anthropometry.waist_vertex_indices
        )

        waist_width = (
            vertices[waist_indices, 0].max()
            - vertices[waist_indices, 0].min()
        )

        chest = self._chest_section(
            rest_vertices,
            context=context,
        )

        hip = self._hip_section(
            rest_vertices,
            context=context,
        )

        neck = self._neck_section(
            rest_vertices,
            rest_bone_poses,
            mesh=mesh,
        )

        upper_arm = self._upper_arm_section(
            rest_vertices,
            rest_bone_poses,
            mesh=mesh,
        )

        forearm = self._smallest_limb_section(
            rest_vertices,
            rest_bone_poses,
            50,
            51,
            0.45,
            mesh=mesh,
        )

        # Same landmarks as before (bones 3/4/5 of the left leg are
        # upperleg02.L, lowerleg01.L, lowerleg02.L), now per side by name.
        thigh_left, thigh_right = (
            self._leg_section(
                rest_vertices,
                rest_bone_poses,
                "upperleg02",
                "lowerleg01",
                side,
                mesh=mesh,
            )
            for side in (".L", ".R")
        )

        calf_left, calf_right = (
            self._leg_section(
                rest_vertices,
                rest_bone_poses,
                "lowerleg01",
                "lowerleg02",
                side,
                mesh=mesh,
            )
            for side in (".L", ".R")
        )

        thigh_values = [
            x
            for x in [
                thigh_left,
                thigh_right,
            ]
            if x is not None
        ]

        calf_values = [
            x
            for x in [
                calf_left,
                calf_right,
            ]
            if x is not None
        ]

        return {
            "height_m": height,
            "mass_kg": mass,
            "bmi": bmi,

            "waist_m": float(
                base["waist_circumference"].item()
            ),

            "waist_width_m": float(
                waist_width
            ),

            "chest_m": (
                chest["circumference"]
                if chest
                else None
            ),

            "chest_width_m": (
                chest["width"]
                if chest
                else None
            ),

            "hip_m": (
                hip["circumference"]
                if hip
                else None
            ),

            "hip_width_m": (
                hip["width"]
                if hip
                else None
            ),

            "hip_z": (
                hip["z"]
                if hip
                else None
            ),

            "neck_m": neck,

            "shoulder_breadth_m": self.shoulder_breadth(
                vertices,
                rest_bone_poses,
            ),

            "upper_arm_m": (
                upper_arm["circumference"]
                if upper_arm
                else None
            ),

            "forearm_m": (
                forearm["circumference"]
                if forearm
                else None
            ),

            "thigh_m": (
                float(np.mean(thigh_values))
                if thigh_values
                else None
            ),

            "thigh_left_m": thigh_left,

            "thigh_right_m": thigh_right,

            "calf_m": (
                float(np.mean(calf_values))
                if calf_values
                else None
            ),

            "calf_left_m": calf_left,

            "calf_right_m": calf_right,

            **self.joint_lengths(rest_bone_poses),
        }


if __name__ == "__main__":
    model = anny.Anny()

    measurements = BodyMeasurements(
        model
    )

    output = model()

    result = measurements.measure(
        output
    )

    def show(label, key, scale=100.0, unit="cm"):
        value = result.get(key)

        if value is None:
            print(f"{label:<13}: N/A")
        else:
            print(f"{label:<13}: {value * scale:.2f} {unit}")

    print()
    print("BODY MEASUREMENTS")
    print("-----------------")

    show("Height", "height_m")
    show("Weight", "mass_kg", 1.0, "kg")
    print(f"{'BMI':<13}: {result['bmi']:.2f}")

    print()
    show("Waist", "waist_m")
    show("Waist width", "waist_width_m")

    print()
    show("Chest", "chest_m")
    show("Chest width", "chest_width_m")

    print()
    show("Hip", "hip_m")
    show("Hip width", "hip_width_m")

    print()
    show("Neck", "neck_m")
    show("Upper arm", "upper_arm_m")
    show("Forearm", "forearm_m")

    print()
    show("Thigh", "thigh_m")
    show("  Left", "thigh_left_m")
    show("  Right", "thigh_right_m")

    print()
    show("Calf", "calf_m")
    show("  Left", "calf_left_m")
    show("  Right", "calf_right_m")

    print()
    show("Shoulder (j)", "shoulder_width_m")
    show("Arm len (j)", "arm_length_m")
    show("Leg len (j)", "leg_length_m")
