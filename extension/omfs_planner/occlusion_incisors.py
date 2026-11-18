"""Incisor relation of the upper arch (or its anterior segment) on the fixed lower arch.

Measured at the midline: overbite (surplomb vertical: how far the upper incisal edge comes down past the
lower one) and overjet (surplomb horizontal: from the lower labial surface to the upper incisal edge).

Glide (segmental optimisation): the anterior segment is moved only in the sagittal plane, without turning
(its angles and midline are the surgeon's). The target overbite sets its height; at that height it is slid
antero-posteriorly to where it just touches the lower arch (overlap at the anterior limit). Done for every
height, that is the incisal guidance: the path keeps the contact, whatever the incisors' slope. Its angle is
read and reported, not imposed."""

from collections import namedtuple
from math import atan2, degrees

import numpy as np
from mathutils import Matrix, Vector

from . import occlusion_seating, occlusion_segments, params, registration

# World points (numpy) and measures, mm. None of them when the incisors can't be found near the midline.
Incisors = namedtuple("Incisors", "upper_tip lower_tip overbite overjet")


def axes(scene):
    """(left, anterior, up), numpy unit vectors: the reference frame's, else the lower block's."""
    return [np.array(v) for v in occlusion_segments.clinical_axes(scene)]


def near_midline(scene, points):
    return points[np.abs(occlusion_seating.midline_offset(scene, points)) < params.SEG_INCISOR_BAND]


def incisal_edge(points, height, lowest):
    """Mean of the few most occlusal points: the lowest for upper teeth, the highest for lower teeth."""
    order = np.argsort(height)
    k = min(params.SEG_INCISOR_TIP_VERTICES, len(points))
    return points[order[:k] if lowest else order[-k:]].mean(axis=0)


def measure(scene, upper_part, lower):
    """Incisors of an upper object (the scan or its anterior segment) on the lower arch, or None."""
    _left, anterior, up = axes(scene)
    upper_points = near_midline(scene, registration.vertex_coords(upper_part))
    lower_points = near_midline(scene, registration.vertex_coords(lower))
    if not len(upper_points) or not len(lower_points):
        return None
    upper_tip = incisal_edge(upper_points, upper_points @ up, lowest=True)
    lower_tip = incisal_edge(lower_points, lower_points @ up, lowest=False)
    overbite = float((lower_tip - upper_tip) @ up)
    level = np.abs((lower_points - upper_tip) @ up) < params.SEG_OVERJET_BAND   # lower labial face, at the edge's height
    labial = (lower_points[level] @ anterior).max() if level.any() else lower_tip @ anterior
    return Incisors(upper_tip, lower_tip, overbite, float(upper_tip @ anterior - labial))


class Contact:
    """Overlap of one upper part against the lower arch for trial translations of the part (nothing moves).
    Only the points that can overlap are tested: of the part, those below the top of the lower teeth (+ the
    deepest descent tested); of the lower arch, those above the bottom of the part (- that descent). Palate,
    gingiva and roots of the crowns are left out (a few times faster, same result)."""

    def __init__(self, part, lower, up, descent):
        margin = params.SEG_GLIDE_RANGE + params.OCC_SEARCH_DISTANCE
        self.part, self.lower = part, lower
        points = registration.vertex_coords(part)
        lower_points = registration.vertex_coords(lower)
        low, high = points.min(axis=0) - margin, points.max(axis=0) + margin
        near = np.all((lower_points >= low) & (lower_points <= high), axis=1)   # only the lower teeth it can meet
        lower_points, lower_areas = lower_points[near], occlusion_seating.vertex_areas(lower)[near]
        h, lower_h = points @ up, lower_points @ up
        slack = max(descent, 0.0) + params.OCC_EXACT_BAND
        keep = h <= lower_h.max() + slack if len(lower_h) else np.zeros(len(h), dtype=bool)
        lower_keep = lower_h >= h.min() - slack
        self.points, self.areas = points[keep], occlusion_seating.vertex_areas(part)[keep]
        self.lower_points, self.lower_areas = lower_points[lower_keep], lower_areas[lower_keep]
        self.matrix = np.array(part.matrix_world)

    def overlap(self, shift):
        """Deepest overlap (mm, robust to lone spikes) with the part moved by shift (world)."""
        search = params.OCC_SEARCH_DISTANCE
        d1 = occlusion_seating.signed_distances(self.points + shift, self.lower, search)[0]
        moved = self.matrix.copy()
        moved[:3, 3] += shift
        d2 = occlusion_seating.signed_distances(self.lower_points, self.part, search, moved)[0]
        return max(occlusion_seating.robust_overlap(d1, self.areas),
                   occlusion_seating.robust_overlap(d2, self.lower_areas))


def contact_position(contact, down, frame, limit):
    """At this height (down: mm moved occlusally), the antero-posterior position (mm, + = forward) where the
    part just touches: the farthest back with its overlap within the limit. Scanned from the back at
    SEG_GLIDE_SCAN, or by the overlap itself when deeper (the overlap is not monotonic: canine slopes), then
    refined between the last position too deep and the first that fits. (position, status), status 'contact', 'free' (no contact even
    SEG_GLIDE_RANGE back: no incisor contact at this height) or 'blocked' (too deep everywhere within
    SEG_GLIDE_RANGE: something else, e.g. the canines, stops the segment at this height)."""
    _left, front, up = frame
    reach = params.SEG_GLIDE_RANGE

    def excess(a):
        return contact.overlap(-up * down + front * a) - limit - params.SEG_GLIDE_PRECISION

    def fits(a):
        return excess(a) <= 0

    back = -reach
    deep = excess(back)
    if deep <= 0:
        return 0.0, 'free'
    while True:   # a point in by `deep` needs at least that much movement to come out: no contact skipped
        if back >= reach - 1e-9:
            return 0.0, 'blocked'
        forward = min(back + max(params.SEG_GLIDE_SCAN, deep), reach)
        deep = excess(forward)
        if deep <= 0:
            break
        back = forward
    while forward - back > params.SEG_GLIDE_PRECISION:
        middle = (back + forward) / 2
        if fits(middle):
            forward = middle
        else:
            back = middle
    return forward, 'contact'


# What a glide did: the overbite before / after (mm), the movement (mm down, mm forward), the guidance
# angle (degrees from the occlusal plane, None if not read), the status of contact_position, and whether the
# target was out of reach (then: as close as the contacts allow).
Glide = namedtuple("Glide", "before after down forward angle status short")


def glide_shift(scene, part, lower, target, limit, read_angle=True):
    """(world translation, Glide) bringing part to the target overbite along the incisal guidance, or
    (None, message) when the incisors can't be measured. A target the contacts don't allow (blocked at that
    height): the closest height that is allowed, found between where the part is and the target."""
    found = measure(scene, part, lower)
    if found is None:
        return None, (f"{part.name} : incisives introuvables à moins de {params.SEG_INCISOR_BAND:g} mm du milieu")
    frame = axes(scene)
    _left, front, up = frame
    down = target - found.overbite
    contact = Contact(part, lower, up, down + params.SEG_GLIDE_SLOPE_STEP)
    forward, status = contact_position(contact, down, frame, limit)
    short = status == 'blocked'
    if short:
        reachable, beyond = 0.0, down                    # search between here and the target
        here, here_status = contact_position(contact, 0.0, frame, limit)
        if here_status == 'blocked':
            return np.zeros(3), Glide(found.overbite, found.overbite, 0.0, 0.0, None, 'blocked', True)
        forward, status = here, here_status
        while abs(beyond - reachable) > params.SEG_GLIDE_HEIGHT_PRECISION:
            middle = (reachable + beyond) / 2
            position, middle_status = contact_position(contact, middle, frame, limit)
            if middle_status == 'blocked':
                beyond = middle
            else:
                reachable, forward, status = middle, position, middle_status
        down = reachable
    angle = None
    if read_angle and status == 'contact':
        step = params.SEG_GLIDE_SLOPE_STEP * (-1 if short or down > 0 else 1)   # towards where it can go
        other, other_status = contact_position(contact, down + step, frame, limit)
        if other_status == 'contact':
            angle = degrees(atan2(abs(step), abs(other - forward)))
    shift = -up * down + front * forward
    return shift, Glide(found.overbite, found.overbite + down, down, forward, angle, status, short)


def glide(scene, part, lower, target, limit):
    """Move the part's block to the target overbite (or as close as the contacts allow), keeping the incisal
    contact. Returns a Glide, or an error message (nothing moved)."""
    shift, result = glide_shift(scene, part, lower, target, limit)
    if shift is None:
        return result
    if result.status == 'blocked':
        return (f"{part.name} : déjà en chevauchement à toutes les positions antéro-postérieures "
                f"(±{params.SEG_GLIDE_RANGE:g} mm) ; vérifiez ses angles et l'assise")
    block = occlusion_segments.block_of(part)
    block.matrix_world = Matrix.Translation(Vector(shift)) @ block.matrix_world
    return result


def describe(result):
    """One line for the surgeon."""
    text = (f"surplomb vertical {result.before:.1f} → {result.after:.1f} mm ; "
            f"{'descendu' if result.down >= 0 else 'monté'} {abs(result.down):.1f} mm, "
            f"{'avancé' if result.forward >= 0 else 'reculé'} {abs(result.forward):.1f} mm")
    if result.angle is not None:
        text += f" ; pente incisive {result.angle:.0f}°"
    if result.short:
        text += f" ; surplomb visé non atteignable : d'autres contacts (ex. canines) bloquent à {result.after:.1f} mm"
    if result.status == 'free':
        text += " ; aucun contact incisif à cette hauteur (position antéro-postérieure gardée)"
    return text
