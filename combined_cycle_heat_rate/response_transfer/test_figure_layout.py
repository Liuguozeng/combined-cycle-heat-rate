"""Recorded-value and interface regression tests; no fitting or resampling."""
from pathlib import Path
import ast
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib import font_manager
from matplotlib.colors import to_rgba
from figure_layout import response_figure, NAMES, BLUE, PINK, INK
import render

HERE = Path(__file__).resolve().parent


class FigureLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = pd.read_csv(HERE / "results/candidate_summaries.csv")
        cls.calibration = pd.read_csv(HERE / "results/historical_calibration.csv")

    def tearDown(self):
        plt.close("all")

    def test_public_entry_delegates_figure_two(self):
        tree = ast.parse((HERE / "render.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "response_figure"]
        self.assertEqual(len(calls), 1)
        self.assertEqual([n.id for n in calls[0].args], ["summary", "cal"])

    def test_bars_intervals_coefficients_and_dimensions(self):
        summary, calibration = self.summary.copy(deep=True), self.calibration.copy(deep=True)
        fig = response_figure(summary, calibration)
        np.testing.assert_allclose(fig.get_size_inches() * 25.4, [178, 106], rtol=0, atol=1e-12)
        a, b = fig.axes
        expected = []
        for index, candidate in enumerate(NAMES):
            for model in ["M1", "M2"]:
                if index < 2 and model == "M2":
                    continue
                expected.append(summary[(summary.scope == "main") &
                                        (summary.model == model) & (summary.candidate == candidate)].iloc[0])
        self.assertEqual(len(a.patches), 8)
        np.testing.assert_array_equal([bar.get_height() for bar in a.patches], [r.mae for r in expected])
        error_lines = [c for c in a.collections if hasattr(c, "get_segments")]
        self.assertEqual(len(error_lines), 8)
        for artist, row in zip(error_lines, expected):
            np.testing.assert_allclose(artist.get_segments()[0][:, 1],
                                       [row.mae_ci95_lower, row.mae_ci95_upper], rtol=0, atol=1e-15)
        points = [c for c in b.collections if isinstance(c, PathCollection)]
        self.assertEqual([len(c.get_offsets()) for c in points], [120, 120])
        for artist, model in zip(points, ["M1", "M2"]):
            values = calibration[calibration.model == model].sort_values(["response_coefficient", "facility_id"])
            np.testing.assert_array_equal(artist.get_offsets()[:, 0], values.response_coefficient)
        pd.testing.assert_frame_equal(summary, self.summary)
        pd.testing.assert_frame_equal(calibration, self.calibration)

    def test_s5_and_all_csv_outputs_remain(self):
        figures = {}
        def collect(fig, output, name):
            figures[name] = fig
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(render, "save", side_effect=collect):
                render.response(HERE / "results", output)
            self.assertEqual(set(figures), {"Figure_2", "Figure_S5"})
            for source, name in [("candidate_summaries.csv", "Figure_2_source_summary.csv"),
                                 ("historical_calibration.csv", "Figure_2_source_coefficients.csv"),
                                 ("stratum_summaries.csv", "Figure_S5_source_strata.csv")]:
                expected = pd.read_csv(HERE / "results" / source).to_csv(index=False, float_format="%.17g")
                self.assertEqual((output / name).read_text(), expected)
            strata = pd.read_csv(HERE / "results/stratum_summaries.csv")
            for ax, model in zip(figures["Figure_S5"].axes[:2], ["M1", "M2"]):
                cells = np.full((3, 3), np.nan)
                for row in strata[(strata.kind == "conditions") & (strata.model == model) &
                                   (strata.candidate == "local_response")].itertuples():
                    temperature, load = map(int, row.stratum.split("|"))
                    cells[temperature - 1, load - 1] = row.minus_historical_mean
                np.testing.assert_array_equal(ax.images[0].get_array(), cells)

    def test_missing_times_new_roman_uses_available_serif(self):
        summary, calibration = self.summary.copy(deep=True), self.calibration.copy(deep=True)
        with patch.object(font_manager, "findfont", side_effect=ValueError("Times New Roman missing")):
            fig = response_figure(summary, calibration)
        self.assertEqual(len(fig.axes[0].patches), 8)
        points = [c for c in fig.axes[1].collections if isinstance(c, PathCollection)]
        self.assertEqual(sum(len(c.get_offsets()) for c in points), 240)
        self.assertEqual(plt.rcParams["font.family"], ["DejaVu Serif"])
        pd.testing.assert_frame_equal(summary, self.summary)
        pd.testing.assert_frame_equal(calibration, self.calibration)

    def test_axis_edges_do_not_inherit_entry_style(self):
        pixels = []
        for inherited_color in ["black", "magenta"]:
            with plt.rc_context({"axes.edgecolor": inherited_color}):
                fig = response_figure(self.summary, self.calibration)
                for ax in fig.axes:
                    for spine in ax.spines.values():
                        self.assertEqual(spine.get_edgecolor(), to_rgba(INK))
                fig.canvas.draw()
                pixels.append(np.asarray(fig.canvas.buffer_rgba()).copy())
                plt.close(fig)
        np.testing.assert_array_equal(*pixels)


if __name__ == "__main__":
    unittest.main()
