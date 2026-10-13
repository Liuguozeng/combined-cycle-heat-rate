"""Physical panel-letter anchors and a render-time geometry receipt."""
import json
from pathlib import Path
from matplotlib.transforms import ScaledTranslation


def panel_label(ax, letter, title=None, title_size=9, content_bounds=None):
    if len(letter) != 1 or not letter.isascii() or not letter.islower():
        raise ValueError('One lowercase ASCII panel letter is required')
    x0, y0, x1, y1 = content_bounds or (0, 0, 1, 1)
    transform = ax.transAxes + ScaledTranslation(0, 2 / 25.4, ax.figure.dpi_scale_trans)
    text = ax.text(x0, y1, letter, transform=transform, ha='left', va='baseline',
                   fontsize=9, fontweight='bold', clip_on=False)
    text.set_gid('panel-letter:' + letter)
    text._panel_anchor = dict(axes=ax, bounds=(x0, y0, x1, y1),
                             kind='content-boundary' if content_bounds else 'plot-frame')
    if title is not None:
        for artist in (ax.title, ax._left_title, ax._right_title):
            artist.set_text('')
        title_transform = ax.transAxes + ScaledTranslation(4 / 25.4, 2 / 25.4,
                                                         ax.figure.dpi_scale_trans)
        ax.text(x0, y1, title, transform=title_transform, ha='left', va='baseline',
                fontsize=title_size, fontweight='bold', clip_on=False)
    return text


def above_frame(ax, text, offset_mm=6, fontsize=8):
    transform = ax.transAxes + ScaledTranslation(0, offset_mm / 25.4,
                                                ax.figure.dpi_scale_trans)
    return ax.text(0, 1, text, transform=transform, va='baseline', fontsize=fontsize,
                   clip_on=False)


def write_panel_geometry(fig, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    rows = []
    for text in fig.findobj():
        if not hasattr(text, '_panel_anchor'):
            continue
        ref = text._panel_anchor
        ax = ref['axes']
        x0, y0, x1, y1 = ref['bounds']
        frame = ax.transAxes.transform([(x0, y0), (x1, y1)]) / fig.dpi * 25.4
        anchor = text.get_transform().transform(text.get_position()) / fig.dpi * 25.4
        box = text.get_window_extent(renderer)
        assert abs(anchor[0] - frame[0, 0]) < 1e-7
        assert abs(anchor[1] - frame[1, 1] - 2) < 1e-7
        assert text.get_fontsize() == 9 and text.get_fontweight() == 'bold'
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
        rows.append(dict(letter=text.get_text(), anchor_kind=ref['kind'],
            frame_left_mm=float(frame[0, 0]), frame_top_mm=float(frame[1, 1]),
            anchor_x_mm=float(anchor[0]), baseline_y_mm=float(anchor[1]),
            offset_x_mm=float(anchor[0] - frame[0, 0]),
            offset_y_mm=float(anchor[1] - frame[1, 1]),
            font_size_pt=9, font_weight='bold', vertical_alignment='baseline',
            text_bbox_mm=(box.extents / fig.dpi * 25.4).tolist(),
            xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim())))
    if rows:
        Path(str(stem) + '.panel_geometry.json').write_text(json.dumps(dict(
            canvas_mm=(fig.get_size_inches() * 25.4).tolist(), panels=rows,
            policy='letter baseline at plot/content left, top + 2 mm; 9 pt bold'), indent=2) + '\n')
