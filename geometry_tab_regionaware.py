
import math
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from geometry_io_regionaware import (
    RegionDef,
    Segment,
    build_geometry_snapshot,
    build_geometry_text,
    parse_geometry,
)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class GeometryTab(QWidget):
    MAIN_HEADERS = ["Name", "Type", "N", "IBC", "IPN1", "IPN2", "PlusReg", "MinusReg"]

    def __init__(self, parent=None):
        super().__init__(parent)
        splitter = QSplitter(Qt.Horizontal)

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        self.canvas = MplCanvas(plot_container)
        self.toolbar = NavigationToolbar(self.canvas, plot_container)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_container)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)

        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load")
        self.btn_save = QPushButton("Save")
        self.btn_validate = QPushButton("Validate")
        self.chk_show_normals = QCheckBox("Show Normals")
        self.chk_show_fills = QCheckBox("Show Region Fills")
        self.chk_show_fills.setChecked(True)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_validate)
        btn_row.addWidget(self.chk_show_normals)
        btn_row.addWidget(self.chk_show_fills)
        btn_row.addStretch(1)
        right_layout.addLayout(btn_row)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self.MAIN_HEADERS))
        self.table.setHorizontalHeaderLabels(self.MAIN_HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        right_layout.addWidget(self.table)

        bottom_row = QHBoxLayout()

        ibc_box = QVBoxLayout()
        self.lbl_ibc = QLabel("IBCS")
        self.table_ibc = QTableWidget()
        ibc_box.addWidget(self.lbl_ibc)
        ibc_box.addWidget(self.table_ibc)

        diel_box = QVBoxLayout()
        self.lbl_diel = QLabel("Dielectrics")
        self.table_diel = QTableWidget()
        diel_box.addWidget(self.lbl_diel)
        diel_box.addWidget(self.table_diel)

        region_box = QVBoxLayout()
        self.lbl_regions = QLabel("Regions")
        self.table_regions = QTableWidget()
        self.table_regions.setColumnCount(3)
        self.table_regions.setHorizontalHeaderLabels(["RegionID", "Material", "Name"])
        region_box.addWidget(self.lbl_regions)
        region_box.addWidget(self.table_regions)

        bottom_row.addLayout(ibc_box, stretch=1)
        bottom_row.addLayout(diel_box, stretch=1)
        bottom_row.addLayout(region_box, stretch=1)
        right_layout.addLayout(bottom_row)
        splitter.addWidget(right_container)

        splitter.setSizes([700, 420])
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(splitter)

        self.btn_load.clicked.connect(self.load_geo)
        self.btn_save.clicked.connect(self.save_geo)
        self.btn_validate.clicked.connect(self.validate_geometry)
        self.chk_show_normals.toggled.connect(self._on_show_normals_toggled)
        self.chk_show_fills.toggled.connect(self._on_show_fills_toggled)

        self.title: str = "Geometry"
        self.segments: List[Segment] = []
        self.region_defs: List[RegionDef] = []
        self.ibcs_entries: List[List[str]] = []
        self.dielectric_entries: List[List[str]] = []
        self.segment_lines: List[Any] = []
        self.segment_base_colors: List[str] = []
        self.normal_artists: List[Any] = []
        self.region_patches: List[Any] = []
        self.issue_rows: Set[int] = set()
        self._populating: bool = False
        self._syncing_selection: bool = False
        self._selected_row: Optional[int] = None
        self._last_ext: str = ".geo"
        self.loaded_path: str = ""

        self.table.itemChanged.connect(self._on_main_table_item_changed)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.canvas.mpl_connect("pick_event", self._on_plot_pick)
        self.canvas.mpl_connect("button_press_event", self._on_plot_button_press)
        self.canvas.mpl_connect("scroll_event", self._on_plot_scroll)

        self._set_equal_column_widths(self.table, enabled=True)
        self._set_equal_column_widths(self.table_ibc, enabled=True)
        self._set_equal_column_widths(self.table_diel, enabled=True)
        self._set_equal_column_widths(self.table_regions, enabled=True)

    # ---------- file I/O ----------
    def load_geo(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Geometry File", "", "Geometry Files (*.geo);;All Files (*)"
        )
        if not fname:
            return
        try:
            text = open(fname, "r").read()
            title, segments, ibcs_entries, dielectric_entries, region_defs = parse_geometry(text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse geometry: {e}")
            return

        self.title = title
        self.segments = segments
        self.region_defs = region_defs
        self.ibcs_entries = ibcs_entries
        self.dielectric_entries = dielectric_entries
        self.loaded_path = os.path.abspath(fname)

        self._populate_all_tables()
        self._rebuild_plot()
        QMessageBox.information(
            self,
            "Loaded",
            f"Loaded {len(self.segments)} segment(s), {len(self.ibcs_entries)} IBC row(s), "
            f"{len(self.dielectric_entries)} dielectric row(s), and {len(self.region_defs)} region row(s).",
        )

    def save_geo(self):
        default_name = f"geometry_out{self._last_ext}"
        fname, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Geometry File", default_name, "Geometry Files (*.geo);;All Files (*)"
        )
        if not fname:
            return
        fname = self._ensure_extension(fname, selected_filter)
        self._last_ext = os.path.splitext(fname)[1].lower()

        ibcs_rows = self._read_small_table(self.table_ibc)
        dielectric_rows = self._read_small_table(self.table_diel)
        region_defs = self._read_region_defs()
        try:
            text = build_geometry_text(self.title, self.segments, ibcs_rows, dielectric_rows, region_defs)
            with open(fname, "w") as f:
                f.write(text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save geometry: {e}")
            return

        self.loaded_path = os.path.abspath(fname)
        QMessageBox.information(self, "Saved", f"Geometry saved to {fname}")

    def get_geometry_snapshot(self) -> Dict[str, Any]:
        snapshot = build_geometry_snapshot(
            self.title,
            self.segments,
            self._read_small_table(self.table_ibc),
            self._read_small_table(self.table_diel),
            self._read_region_defs(),
        )
        snapshot["source_path"] = self.loaded_path
        return snapshot

    # ---------- table population ----------
    def _populate_all_tables(self):
        self._populating = True
        try:
            self.table.clearContents()
            self.table.setRowCount(len(self.segments))
            self.table.setColumnCount(len(self.MAIN_HEADERS))
            self.table.setHorizontalHeaderLabels(self.MAIN_HEADERS)
            for row, seg in enumerate(self.segments):
                props = self._ensure_prop_len(seg.properties, 6)
                values = [
                    seg.name,
                    props[0],
                    props[1],
                    props[3],
                    props[4],
                    props[5],
                    "" if seg.plus_region is None else str(int(seg.plus_region)),
                    "" if seg.minus_region is None else str(int(seg.minus_region)),
                ]
                for col, val in enumerate(values):
                    self.table.setItem(row, col, QTableWidgetItem(val))

            self._populate_small_table(self.table_ibc, self.ibcs_entries, label=self.lbl_ibc, title_prefix="IBCS")
            self._populate_small_table(
                self.table_diel, self.dielectric_entries, label=self.lbl_diel, title_prefix="Dielectrics"
            )
            self._populate_regions_table(self.region_defs)
        finally:
            self._populating = False

    def _populate_small_table(self, table: QTableWidget, rows: List[List[str]], label: QLabel, title_prefix: str):
        col_count = max((len(r) for r in rows), default=0)
        table.clearContents()
        table.setRowCount(len(rows))
        table.setColumnCount(col_count)

        if title_prefix == "IBCS":
            headers = ["Flag", "Z_real", "Z_imag", "Constant"][:col_count]
        elif title_prefix == "Dielectrics":
            headers = ["Flag", "Ep_real", "Ep_imag", "Mu_real", "Mu_imag"][:col_count]
        else:
            headers = [f"Col {i+1}" for i in range(col_count)]
        table.setHorizontalHeaderLabels(headers)
        for r, tokens in enumerate(rows):
            for c, token in enumerate(tokens):
                table.setItem(r, c, QTableWidgetItem(token))
        label.setText(f"{title_prefix} (n={len(rows)})")

    def _populate_regions_table(self, region_defs: List[RegionDef]):
        self.table_regions.clearContents()
        self.table_regions.setRowCount(len(region_defs))
        self.table_regions.setColumnCount(3)
        self.table_regions.setHorizontalHeaderLabels(["RegionID", "Material", "Name"])
        for r, reg in enumerate(region_defs):
            self.table_regions.setItem(r, 0, QTableWidgetItem(str(int(reg.region_id))))
            self.table_regions.setItem(r, 1, QTableWidgetItem(str(int(reg.material_flag))))
            self.table_regions.setItem(r, 2, QTableWidgetItem(str(reg.name or "")))
        self.lbl_regions.setText(f"Regions (n={len(region_defs)})")

    def _ensure_prop_len(self, props: List[str], n: int) -> List[str]:
        if len(props) < n:
            props.extend([""] * (n - len(props)))
        return props

    def _read_small_table(self, table: QTableWidget) -> List[List[str]]:
        rows: List[List[str]] = []
        for r in range(table.rowCount()):
            vals: List[str] = []
            for c in range(table.columnCount()):
                item = table.item(r, c)
                vals.append(item.text().strip() if item else "")
            while vals and vals[-1] == "":
                vals.pop()
            if vals:
                rows.append(vals)
        return rows

    def _read_region_defs(self) -> List[RegionDef]:
        out: List[RegionDef] = []
        for row in self._read_small_table(self.table_regions):
            if len(row) < 2:
                continue
            rid = self._parse_int_token(row[0], 0)
            mat = self._parse_int_token(row[1], 0)
            name = row[2].strip() if len(row) >= 3 else ""
            out.append(RegionDef(region_id=rid, material_flag=mat, name=name))
        return out

    # ---------- plotting ----------
    def _rebuild_plot(self):
        ax = self.canvas.ax
        ax.clear()
        self.segment_lines = []
        self.segment_base_colors = []
        self.issue_rows.clear()
        self._clear_normals()
        self._clear_region_patches()

        if self.chk_show_fills.isChecked():
            self._render_region_fills()

        plot_colors = ["orange", "green", "blue", "gray", "black", "red", "purple", "cyan"]
        for row, seg in enumerate(self.segments):
            props = self._ensure_prop_len(seg.properties, 6)
            itype = props[0]
            try:
                color_index = (int(itype) - 1) % len(plot_colors)
                base_color = plot_colors[color_index]
            except Exception:
                base_color = plot_colors[row % len(plot_colors)]

            xs, ys = self._segment_plot_xy(seg)
            (line2d,) = ax.plot(xs, ys, color=base_color, linewidth=1.5, zorder=3)
            line2d.set_picker(True)
            line2d.set_pickradius(5)
            self.segment_lines.append(line2d)
            self.segment_base_colors.append(base_color)

        ax.set_title(self.title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        self._selected_row = None
        self._refresh_segment_styles()
        self._render_normals()
        self.canvas.draw_idle()

    def _clear_region_patches(self):
        for patch in self.region_patches:
            try:
                patch.remove()
            except Exception:
                pass
        self.region_patches = []

    def _region_color(self, region_id: int, material_flag: int) -> str:
        palette = [
            "#b3d9ff", "#b7f7c4", "#ffe0a3", "#f3b8ff", "#c9c9c9",
            "#ffd1dc", "#d5f5e3", "#f9e79f", "#d6eaf8", "#fadbd8",
        ]
        key = abs(int(region_id) * 131 + int(material_flag) * 17)
        return palette[key % len(palette)]

    def _segment_primitives(self, seg: Segment) -> List[Tuple[float, float, float, float]]:
        count = min(len(seg.x), len(seg.y))
        out: List[Tuple[float, float, float, float]] = []
        for i in range(0, count - 1, 2):
            out.append((seg.x[i], seg.y[i], seg.x[i + 1], seg.y[i + 1]))
        return out

    def _segment_arc_angle_deg(self, seg: Segment) -> float:
        props = self._ensure_prop_len(seg.properties, 6)
        try:
            return float(props[2]) if props[2] else 0.0
        except Exception:
            return 0.0

    def _arc_points(self, x1: float, y1: float, x2: float, y2: float, ang_deg: float, samples: int = 40):
        if abs(ang_deg) < 1e-9:
            return [(x1, y1), (x2, y2)]
        dx = x2 - x1
        dy = y2 - y1
        chord = math.hypot(dx, dy)
        if chord <= 1e-12:
            return [(x1, y1), (x2, y2)]

        ang_rad = math.radians(ang_deg)
        abs_phi = abs(ang_rad)
        sin_half = math.sin(abs_phi * 0.5)
        tan_half = math.tan(abs_phi * 0.5)
        if abs(sin_half) <= 1e-12 or abs(tan_half) <= 1e-12:
            return [(x1, y1), (x2, y2)]

        radius = chord / (2.0 * sin_half)
        h = chord / (2.0 * tan_half)
        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)
        px = -dy / chord
        py = dx / chord
        centers = [(mx + px * h, my + py * h), (mx - px * h, my - py * h)]

        best_center = centers[0]
        best_a0 = 0.0
        best_err = float("inf")
        for cx, cy in centers:
            a0 = math.atan2(y1 - cy, x1 - cx)
            x2_pred = cx + radius * math.cos(a0 + ang_rad)
            y2_pred = cy + radius * math.sin(a0 + ang_rad)
            err = math.hypot(x2_pred - x2, y2_pred - y2)
            if err < best_err:
                best_err = err
                best_center = (cx, cy)
                best_a0 = a0

        cx, cy = best_center
        pts = []
        for i in range(samples + 1):
            t = i / max(samples, 1)
            a = best_a0 + ang_rad * t
            pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
        return pts

    def _segment_plot_xy(self, seg: Segment) -> Tuple[List[float], List[float]]:
        primitives = self._segment_primitives(seg)
        if not primitives:
            return list(seg.x), list(seg.y)
        ang_deg = self._segment_arc_angle_deg(seg)
        xs: List[float] = []
        ys: List[float] = []
        for i, (x1, y1, x2, y2) in enumerate(primitives):
            pts = self._arc_points(x1, y1, x2, y2, ang_deg)
            if i > 0 and pts:
                pts = pts[1:]
            for px, py in pts:
                xs.append(px)
                ys.append(py)
        return xs, ys

    def _infer_segment_region_sides(self, seg: Segment) -> Tuple[Optional[int], Optional[int]]:
        if seg.plus_region is not None or seg.minus_region is not None:
            return seg.plus_region, seg.minus_region
        props = self._ensure_prop_len(seg.properties, 6)
        seg_type = self._parse_int_token(props[0], -1)
        ipn1 = self._parse_int_token(props[4], 0)
        ipn2 = self._parse_int_token(props[5], 0)
        if seg_type == 3:
            return 0, ipn1 if ipn1 > 0 else None
        if seg_type == 5:
            return ipn1 if ipn1 > 0 else None, ipn2 if ipn2 > 0 else None
        if seg_type == 4:
            return ipn1 if ipn1 > 0 else None, None
        if seg_type == 2:
            return None, 0
        return None, None

    def _closed_chain_and_area(self, seg: Segment) -> Tuple[bool, float]:
        primitives = self._segment_primitives(seg)
        if not primitives:
            return False, 0.0
        sx, sy, _, _ = primitives[0]
        _, _, ex, ey = primitives[-1]
        if math.hypot(ex - sx, ey - sy) > 1e-8:
            return False, 0.0
        points = [(sx, sy)] + [(x2, y2) for _, _, x2, y2 in primitives]
        area2 = 0.0
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            area2 += x0 * y1 - x1 * y0
        return True, float(area2)

    def _render_region_fills(self):
        region_map = {int(reg.region_id): reg for reg in self._read_region_defs()}
        for seg in self.segments:
            closed, area2 = self._closed_chain_and_area(seg)
            if not closed:
                continue
            plus_region, minus_region = self._infer_segment_region_sides(seg)
            # left-hand normal = plus side. For CW loops, plus is outside and minus is interior.
            interior_region = minus_region if area2 < 0.0 else plus_region
            if interior_region is None or int(interior_region) <= 0:
                continue
            xs, ys = self._segment_plot_xy(seg)
            if len(xs) < 3 or len(ys) < 3:
                continue
            reg = region_map.get(int(interior_region))
            material_flag = int(reg.material_flag) if reg is not None else int(interior_region)
            color = self._region_color(int(interior_region), material_flag)
            patch = Polygon(list(zip(xs, ys)), closed=True, facecolor=color, edgecolor="none", alpha=0.35, zorder=1)
            self.canvas.ax.add_patch(patch)
            self.region_patches.append(patch)

    def _clear_normals(self):
        for art in self.normal_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.normal_artists = []

    def _render_normals(self):
        self._clear_normals()
        if not self.chk_show_normals.isChecked():
            return
        all_x = [x for seg in self.segments for x in seg.x]
        all_y = [y for seg in self.segments for y in seg.y]
        if not all_x or not all_y:
            return
        diag = max(math.hypot(max(all_x) - min(all_x), max(all_y) - min(all_y)), 1.0)
        arrow_len = 0.04 * diag
        ax = self.canvas.ax
        for row, seg in enumerate(self.segments):
            color = "crimson" if row in self.issue_rows else "magenta"
            for x1, y1, x2, y2 in self._segment_primitives(seg):
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length <= 1e-12:
                    continue
                nx = -dy / length
                ny = dx / length
                mx = 0.5 * (x1 + x2)
                my = 0.5 * (y1 + y2)
                ann = ax.annotate(
                    "",
                    xy=(mx + nx * arrow_len, my + ny * arrow_len),
                    xytext=(mx, my),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.8, "alpha": 0.75},
                    zorder=12,
                )
                self.normal_artists.append(ann)

    def _refresh_segment_styles(self):
        for i, line in enumerate(self.segment_lines):
            if self._selected_row is not None and i == self._selected_row:
                line.set_color("pink")
                line.set_linewidth(2.5)
                line.set_zorder(10)
            elif i in self.issue_rows:
                line.set_color("crimson")
                line.set_linewidth(2.2)
                line.set_zorder(8)
            else:
                base = self.segment_base_colors[i] if i < len(self.segment_base_colors) else "gray"
                line.set_color(base)
                line.set_linewidth(1.5)
                line.set_zorder(3)

    # ---------- UI callbacks ----------
    def _on_main_table_item_changed(self, item: QTableWidgetItem):
        if self._populating:
            return
        row = item.row()
        col = item.column()
        if row < 0 or row >= len(self.segments):
            return
        seg = self.segments[row]
        text = item.text().strip()

        if col == 0:
            seg.name = text
            return

        props = self._ensure_prop_len(seg.properties, 6)
        if col == 1:
            props[0] = text
            seg.seg_type = text or None
        elif col == 2:
            props[1] = text
        elif col == 3:
            props[3] = text
        elif col == 4:
            props[4] = text
        elif col == 5:
            props[5] = text
        elif col == 6:
            seg.plus_region = None if text == "" else self._parse_int_token(text, 0)
        elif col == 7:
            seg.minus_region = None if text == "" else self._parse_int_token(text, 0)

        self._rebuild_plot()

    def _on_table_selection_changed(self):
        if self._syncing_selection:
            return
        row = self.table.currentRow()
        self._apply_selection(row)

    def _apply_selection(self, row: int):
        self._selected_row = row if row is not None and row >= 0 else None
        self._refresh_segment_styles()
        self.canvas.draw_idle()

    def _on_plot_pick(self, event):
        line = getattr(event, "artist", None)
        if not line:
            return
        try:
            row = self.segment_lines.index(line)
        except ValueError:
            return
        self._syncing_selection = True
        try:
            self.table.selectRow(row)
            self._apply_selection(row)
        finally:
            self._syncing_selection = False

    def _on_plot_button_press(self, event):
        if event.inaxes != self.canvas.ax:
            return
        modifier_select = event.button == 1 and (event.key in ("control", "shift"))
        if event.button == 3 or modifier_select:
            idx = self._hit_test(event)
            if idx is not None:
                self._syncing_selection = True
                try:
                    self.table.selectRow(idx)
                    self._apply_selection(idx)
                finally:
                    self._syncing_selection = False

    def _hit_test(self, event) -> Optional[int]:
        for i in reversed(range(len(self.segment_lines))):
            line = self.segment_lines[i]
            contains, _ = line.contains(event)
            if contains:
                return i
        return None

    def _on_plot_scroll(self, event):
        if event.inaxes != self.canvas.ax or event.xdata is None or event.ydata is None:
            return
        base_scale = 1.2 if event.button == "up" else (1 / 1.2)
        self._zoom_at(event.xdata, event.ydata, base_scale)

    def _zoom_at(self, x: float, y: float, scale: float):
        ax = self.canvas.ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        w = (xlim[1] - xlim[0]) / scale
        h = (ylim[1] - ylim[0]) / scale
        ax.set_xlim(x - w / 2, x + w / 2)
        ax.set_ylim(y - h / 2, y + h / 2)
        self.canvas.draw_idle()

    def _on_show_normals_toggled(self, checked: bool):
        _ = checked
        self._render_normals()
        self.canvas.draw_idle()

    def _on_show_fills_toggled(self, checked: bool):
        _ = checked
        self._rebuild_plot()

    # ---------- validation ----------
    def _parse_int_token(self, token: str, default: int = 0) -> int:
        text = (token or "").strip().lower()
        if not text:
            return default
        if text.startswith("fort."):
            text = text.split("fort.", 1)[1]
        try:
            return int(float(text))
        except Exception:
            return default

    def _parse_float_token(self, token: str, default: float = 0.0) -> float:
        text = (token or "").strip()
        if not text:
            return default
        try:
            return float(text)
        except Exception:
            return default

    def _find_fort_file(self, flag: int) -> str:
        name = f"fort.{flag}"
        base_dir = os.path.dirname(self.loaded_path) if self.loaded_path else ""
        for path in (os.path.join(base_dir, name), os.path.join(os.getcwd(), name)):
            if path and os.path.isfile(path):
                return path
        return ""

    def _point_key(self, x: float, y: float, tol: float) -> Tuple[int, int]:
        inv = 1.0 / max(tol, 1e-12)
        return int(round(float(x) * inv)), int(round(float(y) * inv))

    def _segments_intersect(
        self,
        a1: Tuple[float, float],
        a2: Tuple[float, float],
        b1: Tuple[float, float],
        b2: Tuple[float, float],
        tol: float,
    ) -> bool:
        def orient(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        def on_seg(p, q, r):
            return (
                min(p[0], q[0]) - tol <= r[0] <= max(p[0], q[0]) + tol
                and min(p[1], q[1]) - tol <= r[1] <= max(p[1], q[1]) + tol
            )

        o1 = orient(a1, a2, b1)
        o2 = orient(a1, a2, b2)
        o3 = orient(b1, b2, a1)
        o4 = orient(b1, b2, a2)

        if ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)):
            return True
        if abs(o1) <= tol and on_seg(a1, a2, b1):
            return True
        if abs(o2) <= tol and on_seg(a1, a2, b2):
            return True
        if abs(o3) <= tol and on_seg(b1, b2, a1):
            return True
        if abs(o4) <= tol and on_seg(b1, b2, a2):
            return True
        return False

    def validate_geometry(self):
        findings: List[Tuple[str, int, str]] = []
        issue_rows: Set[int] = set()

        ibc_rows = self._read_small_table(self.table_ibc)
        diel_rows = self._read_small_table(self.table_diel)
        reg_defs = self._read_region_defs()
        ibc_flags = {self._parse_int_token(row[0], 0) for row in ibc_rows if row}
        diel_flags = {self._parse_int_token(row[0], 0) for row in diel_rows if row}
        region_map = {int(reg.region_id): int(reg.material_flag) for reg in reg_defs}
        if 0 not in region_map:
            findings.append(("INFO", -1, "Region 0 is implicit ambient air even if not listed in Regions."))

        if len(region_map) != len(reg_defs):
            findings.append(("ERROR", -1, "Regions table contains duplicate region IDs."))

        for reg in reg_defs:
            if reg.region_id <= 0:
                findings.append(("WARN", -1, f"Region {reg.region_id}: non-positive region IDs are reserved/awkward; use positive IDs for physical filled regions."))
            if reg.material_flag < 0:
                findings.append(("ERROR", -1, f"Region {reg.region_id}: material flag must be >= 0."))
            if reg.material_flag > 0 and reg.material_flag not in diel_flags and reg.material_flag <= 50:
                findings.append(("ERROR", -1, f"Region {reg.region_id}: material flag {reg.material_flag} is not defined in Dielectrics."))
            if reg.material_flag > 50 and not self._find_fort_file(reg.material_flag):
                findings.append(("ERROR", -1, f"Region {reg.region_id}: material flag {reg.material_flag} expects missing file fort.{reg.material_flag}."))

        all_points = [(x, y) for seg in self.segments for x, y in zip(seg.x, seg.y)]
        diag = 1.0
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            diag = max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1.0)
        tol = max(1e-8, 1e-6 * diag)

        for row, seg in enumerate(self.segments):
            props = self._ensure_prop_len(seg.properties, 6)
            seg_type = self._parse_int_token(props[0], -1)
            n_panels = self._parse_int_token(props[1], 0)
            ibc = self._parse_int_token(props[3], 0)
            ipn1 = self._parse_int_token(props[4], 0)
            ipn2 = self._parse_int_token(props[5], 0)
            primitives = self._segment_primitives(seg)
            label = f"Row {row + 1} '{seg.name}'"

            if seg_type < 1 or seg_type > 5:
                findings.append(("ERROR", row, f"{label}: invalid TYPE '{props[0]}', expected 1..5."))
                issue_rows.add(row)
            if n_panels == 0:
                findings.append(("WARN", row, f"{label}: N should be non-zero. Positive = explicit panel count, negative = wavelength-based meshing."))
                issue_rows.add(row)
            if not primitives:
                findings.append(("ERROR", row, f"{label}: no line primitives found."))
                issue_rows.add(row)
                continue

            for i, (x1, y1, x2, y2) in enumerate(primitives):
                if math.hypot(x2 - x1, y2 - y1) <= tol:
                    findings.append(("ERROR", row, f"{label}: primitive {i + 1} has near-zero length."))
                    issue_rows.add(row)

            for i in range(len(primitives) - 1):
                _, _, ex, ey = primitives[i]
                nx1, ny1, nx2, ny2 = primitives[i + 1]
                d_start = math.hypot(ex - nx1, ey - ny1)
                d_end = math.hypot(ex - nx2, ey - ny2)
                if d_start > tol:
                    if d_end <= tol:
                        findings.append(("WARN", row, f"{label}: primitive {i + 2} appears reversed relative to previous one."))
                    else:
                        findings.append(("WARN", row, f"{label}: primitive {i + 1} and {i + 2} are not connected."))
                    issue_rows.add(row)

            closed, area2 = self._closed_chain_and_area(seg)
            if closed:
                orient = "CCW" if area2 > 0 else "CW"
                findings.append(("INFO", row, f"{label}: closed chain, orientation {orient}."))
            else:
                findings.append(("WARN", row, f"{label}: open chain (start/end do not close)."))

            if ibc > 0 and ibc <= 50 and ibc not in ibc_flags:
                findings.append(("ERROR", row, f"{label}: IBC flag {ibc} is referenced but not defined in IBCS."))
                issue_rows.add(row)
            if ibc > 50 and not self._find_fort_file(ibc):
                findings.append(("ERROR", row, f"{label}: IBC flag {ibc} expects missing file fort.{ibc}."))
                issue_rows.add(row)

            # Legacy material fields remain supported, but type 3 is now plus=air and minus=IPN1 dielectric.
            if seg_type == 3:
                if ipn1 <= 0:
                    findings.append(("ERROR", row, f"{label}: TYPE 3 requires IPN1 > 0 (dielectric on the minus side)."))
                    issue_rows.add(row)
                if seg.plus_region not in (None, 0):
                    findings.append(("WARN", row, f"{label}: TYPE 3 plus side should normally be ambient air (region 0)."))
                    issue_rows.add(row)
            elif seg_type == 4:
                if ipn1 <= 0:
                    findings.append(("ERROR", row, f"{label}: TYPE 4 requires IPN1 > 0."))
                    issue_rows.add(row)
            elif seg_type == 5:
                if ipn1 <= 0 or ipn2 <= 0:
                    findings.append(("ERROR", row, f"{label}: TYPE 5 requires IPN1 > 0 and IPN2 > 0."))
                    issue_rows.add(row)

            for flag_name, flag in (("IPN1", ipn1), ("IPN2", ipn2)):
                if flag <= 0:
                    continue
                if flag <= 50 and flag not in diel_flags:
                    findings.append(("ERROR", row, f"{label}: dielectric flag {flag_name}={flag} is referenced but not defined."))
                    issue_rows.add(row)
                if flag > 50 and not self._find_fort_file(flag):
                    findings.append(("ERROR", row, f"{label}: {flag_name}={flag} expects missing file fort.{flag}."))
                    issue_rows.add(row)

            plus_region, minus_region = self._infer_segment_region_sides(seg)
            for side_name, reg in (("plus", plus_region), ("minus", minus_region)):
                if reg is None or reg == 0:
                    continue
                if reg not in region_map:
                    findings.append(("WARN", row, f"{label}: {side_name} region {reg} is not defined in Regions; solver will fall back to treating the region ID itself as the material flag."))
                    issue_rows.add(row)

            if closed and seg_type == 3 and area2 > 0:
                findings.append(("WARN", row, f"{label}: TYPE 3 is now plus=air/minus=dielectric. A CCW loop places plus on the interior; use clockwise for a filled dielectric body."))
                issue_rows.add(row)

        # Global topology checks
        global_prims: List[Tuple[int, int, Tuple[float, float, float, float], str]] = []
        for row, seg in enumerate(self.segments):
            for pidx, prim in enumerate(self._segment_primitives(seg)):
                global_prims.append((row, pidx, prim, seg.name))

        endpoint_hits: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
        for row, pidx, (x1, y1, x2, y2), _ in global_prims:
            endpoint_hits.setdefault(self._point_key(x1, y1, tol), []).append((row, pidx, 0))
            endpoint_hits.setdefault(self._point_key(x2, y2, tol), []).append((row, pidx, 1))

        for _key, hits in endpoint_hits.items():
            if len(hits) == 1:
                row = hits[0][0]
                findings.append(("WARN", row, f"Row {row + 1}: dangling endpoint not connected to any other primitive."))
                issue_rows.add(row)

        nprims = len(global_prims)
        for i in range(nprims):
            row_i, pidx_i, prim_i, name_i = global_prims[i]
            x1, y1, x2, y2 = prim_i
            ki0 = self._point_key(x1, y1, tol)
            ki1 = self._point_key(x2, y2, tol)
            for j in range(i + 1, nprims):
                row_j, pidx_j, prim_j, name_j = global_prims[j]
                u1, v1, u2, v2 = prim_j
                kj0 = self._point_key(u1, v1, tol)
                kj1 = self._point_key(u2, v2, tol)
                if ki0 in {kj0, kj1} or ki1 in {kj0, kj1}:
                    continue
                if row_i == row_j and abs(pidx_i - pidx_j) <= 1:
                    continue
                if self._segments_intersect((x1, y1), (x2, y2), (u1, v1), (u2, v2), tol):
                    findings.append(("ERROR", row_i, f"Rows {row_i + 1} ('{name_i}') and {row_j + 1} ('{name_j}') have a non-endpoint primitive intersection."))
                    issue_rows.add(row_i)
                    issue_rows.add(row_j)

        self.issue_rows = issue_rows
        self._refresh_segment_styles()
        self._render_normals()
        self.canvas.draw_idle()

        errors = [m for level, _, m in findings if level == "ERROR"]
        warns = [m for level, _, m in findings if level == "WARN"]
        infos = [m for level, _, m in findings if level == "INFO"]
        summary = f"Validation complete: {len(errors)} error(s), {len(warns)} warning(s), {len(infos)} info message(s)."
        detail = errors + warns + infos
        message = summary + ("\n\n" + "\n".join(detail[:30]) if detail else "\nNo issues found.")
        if len(detail) > 30:
            message += f"\n... ({len(detail)-30} additional message(s))"

        if errors or warns:
            QMessageBox.warning(self, "Geometry Validation", message)
        else:
            QMessageBox.information(self, "Geometry Validation", message)

    # ---------- misc ----------
    def _set_equal_column_widths(self, table: QTableWidget, enabled: bool = True):
        header = table.horizontalHeader()
        if not header:
            return
        header.setSectionResizeMode(QHeaderView.Stretch if enabled else QHeaderView.Interactive)

    def _ensure_extension(self, fname: str, selected_filter: str) -> str:
        root, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext in (".geo", ".txt"):
            return fname
        filt = (selected_filter or "").lower()
        if ".txt" in filt:
            return root + ".txt"
        return root + ".geo"
