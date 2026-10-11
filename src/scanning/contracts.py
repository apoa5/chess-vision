from dataclasses import dataclass, asdict, field
from typing import Literal
import math

SCHEMA_VERSION = "guided_scan_v1"
Resolution = Literal["unobserved", "collecting", "conflicted", "resolved", "user_confirmed"]
Tracking = Literal["uninitialized", "tracking", "degraded", "lost"]

@dataclass(frozen=True)
class BoardProfile:
    board_id: str = "wooden_tournament_39cm"
    square_cm: float = 4.0
    border_cm: float = 3.5
    def __post_init__(self):
        if not self.board_id or not math.isfinite(self.square_cm) or self.square_cm <= 0 or not math.isfinite(self.border_cm) or self.border_cm < 0:
            raise ValueError("Invalid board dimensions or identity")
    @property
    def playing_area_cm(self): return 8 * self.square_cm
    @property
    def total_cm(self): return self.playing_area_cm + 2 * self.border_cm

@dataclass(frozen=True)
class FrameObservation:
    session_id: str
    frame_id: str
    timestamp_ms: float
    board_epoch: int
    orientation_version: int
    geometry_version: int
    width: int
    height: int
    h_board_to_frame: tuple[float, ...]
    source_sha256: str
    orientation_source: Literal["manual", "markings", "marker"]
    schema_version: str = SCHEMA_VERSION
    def __post_init__(self):
        if not self.session_id or not self.frame_id: raise ValueError("Missing frame identity")
        if not math.isfinite(self.timestamp_ms) or self.timestamp_ms < 0: raise ValueError("Invalid local timestamp")
        for v in (self.board_epoch,self.orientation_version,self.geometry_version):
            if type(v) is not int or v < 0: raise ValueError("Invalid version")
        if any(type(v) is not int or v <= 0 for v in (self.width,self.height)): raise ValueError("Invalid source dimensions")
        if len(self.h_board_to_frame)!=9 or not all(math.isfinite(v) for v in self.h_board_to_frame): raise ValueError("Invalid homography")
        if len(self.source_sha256)!=64 or any(c not in "0123456789abcdef" for c in self.source_sha256): raise ValueError("Invalid source hash")
        if self.orientation_source not in ("manual","markings","marker"): raise ValueError("Invalid orientation provenance")
        if self.schema_version != SCHEMA_VERSION: raise ValueError("Unsupported schema")
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class ScanSnapshot:
    session_id: str
    board_epoch: int = 0
    tracking: Tracking = "uninitialized"
    squares: dict[str, Resolution] = field(default_factory=lambda: {f"{f}{r}": "unobserved" for f in "abcdefgh" for r in range(1,9)})
    schema_version: str = SCHEMA_VERSION
    def __post_init__(self):
        expected={f"{f}{r}" for f in "abcdefgh" for r in range(1,9)}
        if not self.session_id or type(self.board_epoch) is not int or self.board_epoch < 0: raise ValueError("Invalid session")
        if self.schema_version != SCHEMA_VERSION or self.tracking not in ("uninitialized","tracking","degraded","lost"): raise ValueError("Invalid state")
        if set(self.squares)!=expected or any(v not in ("unobserved","collecting","conflicted","resolved","user_confirmed") for v in self.squares.values()): raise ValueError("Expected 64 valid square states")
    def to_dict(self): return asdict(self)
