Registered grid-footprint comparisons
====================================

The GitHub edition includes geometry records, CSV tables, code and SVG figures.
Separate source composites and PDF/PNG exports are retained in the local
research package. Geometry figures can be rebuilt from the public JSON.

Figures
-------
FigR_placement_registered: two selected pairs passing registration diagnostics.
FigR_placement_exploratory: three additional selected comparisons for inspection.
FigR_placement_all_five: all five comparisons on one page.

PDF and SVG retain editable vector geometry and text. PNG exports are 600 dpi.
The first two figures are 180 x 95 mm and 180 x 77 mm. For a reply letter,
insert at 160 mm width (heights 84.4 mm and 68.4 mm, respectively).

Interpretation
--------------
The geometric comparisons use the independently registered coordinate frame,
with no additional adjustment of either footprint to improve its overlap.
The two diagnostic-passing pairs have footprint IoUs of 94.12% and 84.40%.
The three exploratory pairs have IoUs of 88.93%, 89.47%, and 85.77%.
The displayed geometry is translated and uniformly normalized by the reference
side length; it is not otherwise changed. Both footprints are shown in full.
These are selected photo pairs, including two dates for G3-2. The selection
history and the original supplied scores remain in the source records.

Reconstruction
--------------
The supplied JSON files contain selected-angle footprint coordinates, nine
reference cell centers, the angle settings, registered baseline scores, and
the independent after-to-before registration matrix. The v11 layout uses an
integer outer side length and floor(side / 3) as the cell pitch. The outer
reference center is recovered by correcting the corresponding center offset.
The recorded footprint rotations are inverted about their recorded centers.
A double-precision convex-polygon intersection implementation recomputes both
the registered and selected-angle scores. All differences are less than
0.0001 percentage point, consistent with source polygon floating-point precision.
The archived coordinates and settings in sources/pair_XX_source.json are
sufficient to reproduce these geometric comparisons. They record the original
v11 layout; the figure rebuild does not run the final-B segmenter or planner.

Rebuild
-------
Python dependencies: numpy, matplotlib.
Run from this folder:
    python build_placement_figures.py

data/registered_geometry.json contains restored polygons, registration matrices,
display transforms, plan status, and recomputation checks.
data/overlap_summary.csv contains the numeric comparison.
sources/pair_XX_source.json preserves the supplied geometry records.

Photographic material
---------------------
The five supplied composite JPEGs are retained byte-for-byte in the local
sources/supplied_photo_composites directory. Their original labels and angle-optimized
overlay state are unchanged. They are archival source composites, not the
photographic counterpart of the new registered-geometry figures.
The separate before/after source photographs and the original registration
implementation are not available in the accessible project files. Their
recorded source filenames are retained as portable archival references in the JSON records.
A new registered photo overlay requires those separate source photographs.

Scope
-----
The analysis compares inferred grid-footprint areas in two dimensions.
It does not measure the delivered spatial exposure or individual electrode
targeting. The three exploratory pairs are separated from the two pairs that
passed the source registration diagnostics. The two 26 September plan
assignments remain provisional in the supplied records.
