"""Static configuration for the TCS module.

Scoped to a single object class (oil-palm tree / "sawit") so it does not
overlap with the separate Automatic Feature Extraction (AFE) application in
the Jagad View series.
"""
from __future__ import annotations

DEFAULT_TILE_SIZE = 640
DEFAULT_TILE_OVERLAP = 96
DEFAULT_CONFIDENCE = 0.3
DEFAULT_IOU = 0.25

# Only used by PlaceholderTreeDetector: approximate oil-palm canopy radius and
# minimum center-to-center spacing, in source-raster pixels. Tune these to the
# imagery's ground sample distance when no trained model is available yet.
DEFAULT_CROWN_RADIUS_PX = 18
DEFAULT_CROWN_SPACING_PX = 40

# Used by the optional "Cari Lahan Kosong" (find empty planting land)
# feature: a candidate grid slot counts as "already occupied" (not empty)
# when an existing tree sits within GAP_TOLERANCE_RATIO * estimated_spacing
# of it. 0.5 means within half the ideal spacing, which tolerates the normal
# jitter of real plantations without flagging slots right next to a tree
# that is merely slightly off-grid as empty.
GAP_TOLERANCE_RATIO = 0.5
