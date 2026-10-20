# Figure sources

`rebuild_figures.py` rebuilds the replacement diagrams, internal segmentation
plots and tables, and wound-area plots from the numerical files in the Git
repository. It preserves the supplied `FigureS22_placement_overlays.svg`, whose
five photographs are embedded in the SVG. The default command does not require
the separate area photographs, masks, overlays or placement source JPEGs:

```bash
python plotting/rebuild_figures.py
python plotting/verify_figures.py
```

Area statistics and Fig. 2K/L are recomputed from `measurements.csv` using
`reproduce_area.py --plots-only`. This checks the recorded numerical area
conversion and regenerates the plots; it does not verify original photographs,
mask pixel counts or photographic processing.

For the complete local archive, including the area review pages and placement
source photographs, use:

```bash
python plotting/rebuild_figures.py --include-local-photos
```

That option checks for the required photographic inputs before rendering.
The source photographs and companion PDF exports remain in the authors' local
archive and are not among the GitHub Release attachments.

The internal figures use the six W70 `aspp_small_rates` runs and the existing
eight baseline models. `internal/data/benchmark_per_seed.csv` is the common
54-row source table. Class metrics and confusion matrices are also supplied as
long-form CSV tables; uncertainty is sample standard deviation across seeds.

The graph in `architecture/architecture.reviewed.yaml` is the final B
256-pixel model. `architecture_source_excerpt.txt` records the constructor and
forward operations used to check the graph. The model itself is provided in
the repository's training source directory. The drawing script does not load
weights or execute a neural network.

The workflow and architecture retain the previous editable vector layout with
updated labels and tensor sizes. Existing experimental photographs are only
used in the placement figure, whose reconstruction and processing records
remain under `../analysis/placement/`.

The complete replacement map is in
[`../figures/FIGURE_REPLACEMENT_GUIDE.md`](../figures/FIGURE_REPLACEMENT_GUIDE.md).
Arimo is bundled under `internal/fonts/` with its SIL Open Font License.
Diagrams and photographic labels use DejaVu Sans.
