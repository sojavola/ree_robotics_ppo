from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter, label, binary_dilation


class AdvancedMineralGenerator:
    """Generate realistic REE mineral maps.

    Produces small, spatially dispersed deposits (not large uniform regions).
    Each REE type has its own rarity, deposit size, and depth bias.
    """

    def __init__(
        self, width: int, height: int, seed: Optional[int] = None
    ) -> None:
        self.width = width
        self.height = height
        # Local RNG — avoids corrupting the global NumPy random state.
        self.rng = np.random.default_rng(seed)

        # Geological properties per REE type:
        #   n_deposits   : number of deposits placed on the map
        #   radius_range : (min, max) deposit radius in cells
        #   peak_conc    : maximum concentration at deposit centre [0, 1]
        #   depth_bias   : 0 = surface, 1 = deep (affects underground layers)
        self.mineral_types: Dict[str, Dict] = {
            'REE_Oxides': {
                'color': [1.0, 0.4, 0.0],
                'n_deposits': 5,
                'radius_range': (7, 13),
                'peak_conc': 0.88,
                'depth_bias': 0.3,
            },
            'REE_Silicates': {
                'color': [0.1, 0.75, 0.2],
                'n_deposits': 7,
                'radius_range': (5, 10),
                'peak_conc': 0.78,
                'depth_bias': 0.5,
            },
            'REE_Phosphates': {
                'color': [0.55, 0.1, 0.8],
                'n_deposits': 4,
                'radius_range': (5, 9),
                'peak_conc': 0.92,
                'depth_bias': 0.2,
            },
            'REE_Carbonates': {
                'color': [0.9, 0.75, 0.1],
                'n_deposits': 6,
                'radius_range': (5, 11),
                'peak_conc': 0.72,
                'depth_bias': 0.4,
            },
        }

    # ------------------------------------------------------------------ #
    #  MAP GENERATION                                                      #
    # ------------------------------------------------------------------ #

    def generate_geological_map(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate the surface mineral map.  Same seed produces the same map.

        BUG FIX: the former default was seed=0 which always re-initialised the
        RNG to zero, producing an identical map every episode regardless of the
        seed passed to the constructor.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mineral_map = np.zeros(
            (self.height, self.width, len(self.mineral_types)), dtype=np.float32
        )
        for idx, (_, props) in enumerate(self.mineral_types.items()):
            mineral_map[:, :, idx] = self._place_deposits(props)
        return mineral_map

    def _place_deposits(self, props: Dict) -> np.ndarray:
        """Place N deposits distributed across the map via a zone grid.

        The map is divided into zones to guarantee spatial dispersion and
        prevent all deposits from clustering in one corner.
        """
        layer = np.zeros((self.height, self.width), dtype=np.float32)

        n = props['n_deposits']
        r_min, r_max = props['radius_range']
        peak = props['peak_conc']
        margin = r_max + 3

        ys = np.arange(self.height)
        xs = np.arange(self.width)
        xx, yy = np.meshgrid(xs, ys)

        n_zones_x = max(2, int(np.ceil(np.sqrt(n * self.width / self.height))))
        n_zones_y = max(2, int(np.ceil(n / n_zones_x)) + 1)
        zone_w = (self.width - 2 * margin) / n_zones_x
        zone_h = (self.height - 2 * margin) / n_zones_y

        all_zones = [(zi, zj) for zi in range(n_zones_x) for zj in range(n_zones_y)]
        self.rng.shuffle(all_zones)
        chosen_zones = all_zones[:n]

        for zi, zj in chosen_zones:
            x_lo = int(margin + zi * zone_w)
            x_hi = int(margin + (zi + 1) * zone_w)
            y_lo = int(margin + zj * zone_h)
            y_hi = int(margin + (zj + 1) * zone_h)

            # Fix: safe randint — avoids crash when zone is narrower than margin
            cx_lo = max(margin, x_lo)
            cx_hi = max(cx_lo + 1, min(self.width - margin, x_hi) + 1)
            cy_lo = max(margin, y_lo)
            cy_hi = max(cy_lo + 1, min(self.height - margin, y_hi) + 1)

            cx = int(self.rng.integers(cx_lo, cx_hi))
            cy = int(self.rng.integers(cy_lo, cy_hi))
            r = float(self.rng.uniform(r_min, r_max))

            # Slightly elliptical deposits avoid straight-line artefacts
            angle = float(self.rng.uniform(0, np.pi))
            ratio = float(self.rng.uniform(0.60, 0.95))

            dx = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
            dy = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
            dist2 = dx ** 2 + (dy / ratio) ** 2

            peak_actual = peak * float(self.rng.uniform(0.70, 1.0))
            sigma = r / 2.5
            gaussian = peak_actual * np.exp(-dist2 / (2 * sigma ** 2))
            mask = dist2 <= (r * 1.1) ** 2

            noise = 1.0 + self.rng.uniform(-0.06, 0.06, size=(self.height, self.width))
            layer = np.maximum(layer, gaussian * noise * mask)

        return np.clip(layer, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    #  UNDERGROUND LAYERS                                                  #
    # ------------------------------------------------------------------ #

    def generate_underground_layers(
        self, surface_mineral_map: np.ndarray, num_layers: int = 3
    ) -> List[np.ndarray]:
        """Generate depth layers derived from the surface map.

        Minerals with depth_bias < 0.5 attenuate with depth;
        those with depth_bias >= 0.5 intensify.
        """
        underground_layers = []
        for depth in range(num_layers):
            depth_factor = (depth + 1) / num_layers
            layer_map = surface_mineral_map.copy()
            for midx, (_, props) in enumerate(self.mineral_types.items()):
                db = props['depth_bias']
                if db < 0.5:
                    reduction = (1.0 - db) * depth_factor
                    layer_map[:, :, midx] *= (1.0 - reduction)
                else:
                    enhancement = (db - 0.5) * depth_factor * 2.0
                    layer_map[:, :, midx] *= (1.0 + enhancement)
            underground_layers.append(np.clip(layer_map, 0.0, 1.0))
        return underground_layers

    # ------------------------------------------------------------------ #
    #  CLUSTER DETECTION                                                   #
    # ------------------------------------------------------------------ #

    def detect_mineral_clusters(
        self,
        mineral_map: np.ndarray,
        mineral_idx: int,
        min_samples: int = 3,
        eps: float = 2.5,
    ) -> List[Dict]:
        """Detect deposit clusters via morphological connected components."""
        layer = mineral_map[:, :, mineral_idx]
        mask = layer > 0.15

        if np.count_nonzero(mask) < min_samples:
            return []

        r = int(np.ceil(eps))
        cy, cx = np.ogrid[-r:r + 1, -r:r + 1]
        struct = (cy ** 2 + cx ** 2) <= eps ** 2
        dilated = binary_dilation(mask, structure=struct)
        labeled_map, num_features = label(dilated)

        clusters = []
        for cid in range(1, num_features + 1):
            pts = np.argwhere(mask & (labeled_map == cid))
            if len(pts) < min_samples:
                continue
            clusters.append({
                'points': pts,
                'center': np.mean(pts, axis=0),
                'size': len(pts),
                'max_concentration': float(np.max(layer[pts[:, 0], pts[:, 1]])),
            })
        return clusters
