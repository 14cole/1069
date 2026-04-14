
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Segment:
    name: str
    seg_type: Optional[str]
    properties: List[str]
    x: List[float]
    y: List[float]
    plus_region: Optional[int] = None
    minus_region: Optional[int] = None


@dataclass
class RegionDef:
    region_id: int
    material_flag: int
    name: str = ""


def _parse_region_tokens(tokens: List[str]) -> Tuple[Optional[int], Optional[int]]:
    plus_region: Optional[int] = None
    minus_region: Optional[int] = None
    if len(tokens) >= 1 and str(tokens[0]).strip():
        plus_region = int(float(tokens[0]))
    if len(tokens) >= 2 and str(tokens[1]).strip():
        minus_region = int(float(tokens[1]))
    return plus_region, minus_region


def parse_geometry(
    text: str,
) -> Tuple[str, List[Segment], List[List[str]], List[List[str]], List[RegionDef]]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    title = "Geometry"
    segments: List[Segment] = []
    ibcs_entries: List[List[str]] = []
    dielectric_entries: List[List[str]] = []
    region_defs: List[RegionDef] = []

    state = "segments"
    current_name: Optional[str] = None
    current_type: Optional[str] = None
    current_props: List[str] = []
    cur_x: List[float] = []
    cur_y: List[float] = []
    current_plus_region: Optional[int] = None
    current_minus_region: Optional[int] = None

    def flush_segment() -> None:
        nonlocal current_plus_region, current_minus_region
        if current_name is not None:
            segments.append(
                Segment(
                    name=current_name,
                    seg_type=current_type,
                    properties=current_props[:],
                    x=cur_x[:],
                    y=cur_y[:],
                    plus_region=current_plus_region,
                    minus_region=current_minus_region,
                )
            )
        current_plus_region = None
        current_minus_region = None

    for raw in lines:
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        low = ln.lower()
        if low.startswith("title"):
            title = ln.split(":", 1)[1].strip() or title
            continue
        if state == "segments" and low.startswith("ibcs:"):
            flush_segment()
            state = "ibcs"
            continue
        if low.startswith("dielectrics:"):
            if state == "segments":
                flush_segment()
            state = "dielectrics"
            continue
        if low.startswith("regions:"):
            if state == "segments":
                flush_segment()
            state = "regions"
            continue

        if state == "segments":
            if low.startswith("segment:"):
                flush_segment()
                parts = ln.split(":", 1)[1].strip().split()
                if not parts:
                    current_name, current_type = "Unnamed", None
                elif len(parts) == 1:
                    current_name, current_type = parts[0], None
                else:
                    current_name, current_type = parts[0], parts[1]
                current_props = []
                cur_x.clear()
                cur_y.clear()
                current_plus_region = None
                current_minus_region = None
                continue
            if low.startswith("properties:"):
                current_props = ln.split(":", 1)[1].strip().split()
                continue
            if low.startswith("region_sides:") or low.startswith("regionsides:") or low.startswith("regions:"):
                toks = ln.split(":", 1)[1].strip().split()
                current_plus_region, current_minus_region = _parse_region_tokens(toks)
                continue

            tokens = ln.split()
            if len(tokens) != 4:
                raise ValueError(f"Geometry line must have 4 numbers, got {len(tokens)} {ln}")
            try:
                x1, y1, x2, y2 = map(float, tokens)
            except ValueError as exc:
                raise ValueError(f"Geometry line must contain valid numbers: {ln}") from exc
            cur_x.extend([x1, x2])
            cur_y.extend([y1, y2])

        elif state == "ibcs":
            tokens = ln.split()
            if tokens:
                ibcs_entries.append(tokens)
        elif state == "dielectrics":
            tokens = ln.split()
            if tokens:
                dielectric_entries.append(tokens)
        elif state == "regions":
            tokens = ln.split()
            if not tokens:
                continue
            if len(tokens) < 2:
                raise ValueError(f"Regions row must have at least region_id and material_flag: {ln}")
            try:
                region_id = int(float(tokens[0]))
                material_flag = int(float(tokens[1]))
            except ValueError as exc:
                raise ValueError(f"Invalid Regions row: {ln}") from exc
            name = " ".join(tokens[2:]).strip()
            region_defs.append(RegionDef(region_id=region_id, material_flag=material_flag, name=name))

    if state == "segments":
        flush_segment()

    return title, segments, ibcs_entries, dielectric_entries, region_defs


def build_geometry_text(
    title: str,
    segments: List[Segment],
    ibcs_entries: List[List[str]],
    dielectric_entries: List[List[str]],
    region_defs: Optional[List[RegionDef]] = None,
) -> str:
    lines: List[str] = [f"Title: {title}"]
    for seg in segments:
        props = list(seg.properties)
        effective_type = props[0] if props and str(props[0]).strip() else seg.seg_type
        if effective_type:
            lines.append(f"Segment: {seg.name} {effective_type}")
        else:
            lines.append(f"Segment: {seg.name}")

        if len(props) < 6:
            props.extend([""] * (6 - len(props)))
        lines.append("properties: " + " ".join(p if p is not None else "" for p in props))

        if seg.plus_region is not None or seg.minus_region is not None:
            plus = "" if seg.plus_region is None else str(int(seg.plus_region))
            minus = "" if seg.minus_region is None else str(int(seg.minus_region))
            lines.append(f"region_sides: {plus} {minus}".rstrip())

        if len(seg.x) != len(seg.y) or len(seg.x) % 2 != 0:
            raise ValueError(f"Segment {seg.name} has mismatched or odd number of coordinates.")
        for i in range(0, len(seg.x), 2):
            x1, y1, x2, y2 = seg.x[i], seg.y[i], seg.x[i + 1], seg.y[i + 1]
            lines.append(f"{x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f}")

    lines.append("IBCS:")
    for row in ibcs_entries:
        lines.append(" ".join(row))
    lines.append("Dielectrics:")
    for row in dielectric_entries:
        lines.append(" ".join(row))

    region_defs = list(region_defs or [])
    if region_defs:
        lines.append("Regions:")
        for reg in region_defs:
            tail = f" {reg.name}" if reg.name else ""
            lines.append(f"{int(reg.region_id)} {int(reg.material_flag)}{tail}")
    return "\n".join(lines) + "\n"


def build_geometry_snapshot(
    title: str,
    segments: List[Segment],
    ibcs_entries: List[List[str]],
    dielectric_entries: List[List[str]],
    region_defs: Optional[List[RegionDef]] = None,
) -> Dict[str, Any]:
    segments_payload = []
    for seg in segments:
        point_pairs = []
        for i in range(0, min(len(seg.x), len(seg.y)), 2):
            if i + 1 >= len(seg.x) or i + 1 >= len(seg.y):
                break
            point_pairs.append(
                {
                    "x1": seg.x[i],
                    "y1": seg.y[i],
                    "x2": seg.x[i + 1],
                    "y2": seg.y[i + 1],
                }
            )
        props = list(seg.properties)
        effective_type = props[0] if props and str(props[0]).strip() else seg.seg_type
        payload = {
            "name": seg.name,
            "seg_type": effective_type,
            "properties": props,
            "point_pairs": point_pairs,
        }
        if seg.plus_region is not None:
            payload["plus_region"] = int(seg.plus_region)
        if seg.minus_region is not None:
            payload["minus_region"] = int(seg.minus_region)
        segments_payload.append(payload)

    region_defs = list(region_defs or [])
    region_payload = [
        {
            "region_id": int(reg.region_id),
            "material_flag": int(reg.material_flag),
            "name": str(reg.name or ""),
        }
        for reg in region_defs
    ]

    return {
        "title": title,
        "segment_count": len(segments),
        "segments": segments_payload,
        "ibcs": [list(row) for row in ibcs_entries],
        "dielectrics": [list(row) for row in dielectric_entries],
        "regions": region_payload,
        "region_count": len(region_payload),
    }
