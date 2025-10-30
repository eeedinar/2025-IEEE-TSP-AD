"""
NMF (Non-negative Matrix Factorization) Processor
=================================================

Handles NMF transformation with flexible rank selection options.

"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.decomposition import NMF
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class NMFProcessor:
    """
    NMF processor with flexible rank selection options.
    
    Supports:
    - 'auto': Evaluate all ranks from 1 to n_features and automatic selection via penalized Kneedle algorithm
    - List of integers: Evaluate specific ranks and select best
    - Integer: Fixed number of components
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NMF processor from configuration."""
        # Updated to match config parameter names (no nmf_ prefix)
        self.n_components = config.get('n_components', 'auto')
        self.max_iter = config.get('max_iter', 1000000)
        self.tol = float(config.get('tolerance', 1e-4))
        self.init = config.get('init', 'nndsvd')
        self.random_state = config.get('random_state', 42)
        
        self.nmf = None
        self.selected_rank = None
        self.rank_analysis = {}
        
    def fit_transform(self, X: np.ndarray, plot_path: Optional[str] = None) -> np.ndarray:
        """Fit NMF and transform data."""
        # Determine ranks to evaluate
        ranks_to_evaluate = self._get_ranks_to_evaluate(X.shape)
        
        # Select optimal rank
        if len(ranks_to_evaluate) == 1:
            self.selected_rank = ranks_to_evaluate[0]
            logger.info(f"Using fixed rank: {self.selected_rank}")
        else:
            self.selected_rank = self._select_optimal_rank(X, ranks_to_evaluate, plot_path)
            logger.info(f"Selected optimal rank: {self.selected_rank} from {len(ranks_to_evaluate)} candidates")
        
        # Fit final NMF model
        self.nmf = NMF(
            n_components=self.selected_rank,
            init=self.init,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol
        )
        
        return self.nmf.fit_transform(X) # W = model.fit_transform(X) H = model.components_
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted NMF."""
        if self.nmf is None:
            raise ValueError("NMF not fitted. Call fit_transform first.")
        return self.nmf.transform(X)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save fitted NMF processor to disk."""
        if self.nmf is None:
            raise ValueError("Cannot save unfitted NMF processor")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save complete state including metadata
        joblib.dump({
            'nmf_model': self.nmf,
            'metadata': self.get_metadata()
        }, filepath)
        
        logger.info(f"Saved NMF processor to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'NMFProcessor':
        """Load fitted NMF processor from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"NMF processor file not found: {filepath}")
        
        data = joblib.load(filepath)
        metadata = data['metadata']
        
        # Recreate configuration from metadata (updated parameter names)
        config = {
            'n_components': metadata['n_components_config'],
            'max_iter': metadata['parameters']['max_iter'],
            'tolerance': metadata['parameters']['tolerance'],
            'init': metadata['parameters']['init'],
            'random_state': metadata['parameters']['random_state']
        }
        
        # Create instance and restore state
        instance = cls(config)
        instance.nmf = data['nmf_model']
        instance.selected_rank = metadata['selected_rank']
        instance.rank_analysis = metadata.get('rank_analysis', {})
        
        logger.info(f"Loaded NMF processor from {filepath} (rank={instance.selected_rank})")
        return instance
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get NMF metadata for tracking and saving."""
        return {
            'n_components_config': self.n_components,
            'selected_rank': self.selected_rank,
            'parameters': {
                'max_iter': self.max_iter,
                'tolerance': self.tol,
                'init': self.init,
                'random_state': self.random_state
            },
            'rank_analysis': self.rank_analysis
        }
    
    def _get_ranks_to_evaluate(self, data_shape: Tuple[int, int]) -> List[int]:
        """Determine which ranks to evaluate based on configuration."""
        max_possible_rank = min(data_shape[0], data_shape[1])
        
        if isinstance(self.n_components, int):
            return [min(self.n_components, max_possible_rank)]
        elif isinstance(self.n_components, list):
            valid_ranks = [r for r in self.n_components if 1 <= r <= max_possible_rank]
            if not valid_ranks:
                raise ValueError(f"No valid ranks in {self.n_components} for data shape {data_shape}")
            return sorted(valid_ranks)
        elif self.n_components == 'auto':
            return list(range(1, max_possible_rank + 1))    
        else:
            raise ValueError(f"Invalid n_components: {self.n_components}")
    
    def _select_optimal_rank(
        self, 
        X: np.ndarray, 
        ranks: List[int], 
        plot_path: Optional[str] = None
    ) -> int:
        """Select optimal rank from candidates."""
        # Compute reconstruction errors
        errors = []
        for r in ranks:
            nmf = NMF(
                n_components=r,
                init=self.init,
                random_state=self.random_state,
                max_iter=self.max_iter,
                tol=self.tol
            )
            W = nmf.fit_transform(X)
            H = nmf.components_
            error = np.linalg.norm(X - W @ H, 'fro') ** 2
            errors.append(error)
            
        # Store analysis
        self.rank_analysis = {
            'evaluated_ranks': ranks,
            'reconstruction_errors': errors
        }

        # Use penalized Kneedle algorithm
        return self.select_rank_with_penalized_kneedle_and_inflection(ranks, errors, plot_path)
    
    def select_rank_with_penalized_kneedle_and_inflection(
        self, 
        ranks: List[int], 
        errors: List[float], 
        plot_path: Optional[str] = None
    ) -> int:
        """Select optimal rank using penalized Kneedle algorithm."""
        x_raw = np.array(ranks)
        y_raw = np.array(errors)
        
        # Normalize data
        x = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min()) if x_raw.max() > x_raw.min() else x_raw
        y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min()) if y_raw.max() > y_raw.min() else y_raw
        
        # Kneedle: find maximum distance from diagonal
        linear_line = 1 - x
        distances = np.abs(y - linear_line)
        kneedle_idx = np.argmax(distances)
        r_kneedle_norm = x[kneedle_idx]
        
        # Interpolated rank selection with ceiled lookup
        r_kneedle = int(x_raw[np.where(x >= r_kneedle_norm)[0][0]]) if len(np.where(x >= r_kneedle_norm)[0]) > 0 else ranks[kneedle_idx]
        
        # Initialize spline-related variables
        spline = None
        inflection_candidates = []
        r_elbow = None
        
        # Try to find inflection point via spline
        if len(ranks) >= 4:
            try:
                spline = UnivariateSpline(x, y, k=3, s=1e-10)
                x_dense = np.linspace(0, 1, 500)
                d2 = spline.derivative(n=2)(x_dense)
                
                # Find sign changes in second derivative
                sign_changes = np.where(np.diff(np.sign(d2)))[0]
                inflection_candidates = x_dense[sign_changes]
                
                if len(inflection_candidates) > 0:
                    # # Choose inflection point closest to median
                    r_elbow_norm = min(inflection_candidates, key=lambda r: abs(r - np.median(x)))
                    
                    # # Choose inflection point from last two candidates
                    # if len(inflection_candidates) >= 2:
                    #     # Take the second to last inflection point
                    #     r_elbow_norm = inflection_candidates[-2]
                    # else:
                    #     # If only one candidate, use it
                    #     r_elbow_norm = inflection_candidates[-1]

                    # Map back to original rank
                    r_elbow = int(x_raw[np.where(x >= r_elbow_norm)[0][0]]) if len(np.where(x >= r_elbow_norm)[0]) > 0 else None
            except Exception as e:
                logger.debug(f"Spline fitting failed: {e}")
        
        # Final selected rank
        r_star = r_elbow if r_elbow is not None else r_kneedle
        
        # Store analysis results for potential reuse
        self.rank_analysis.update({
            'evaluated_ranks': ranks,
            'reconstruction_errors': errors,
            'normalized_x': x,
            'normalized_y': y,
            'distances': distances,
            'linear_line': linear_line,
            'kneedle_idx': kneedle_idx,
            'r_kneedle': r_kneedle,
            'r_star': r_star,
            'spline': spline,
            'inflection_candidates': inflection_candidates
        })
        
        # Create plot if requested
        if plot_path:
            self._plot_kneedle_analysis(plot_path)
        
        return r_star

    def _plot_kneedle_analysis(self, plot_path):
        """Create Kneedle analysis visualization using stored analysis data."""
        # Extract data from stored analysis
        analysis = self.rank_analysis
        x = analysis['normalized_x']
        y = analysis['normalized_y']
        ranks = analysis['evaluated_ranks']
        distances = analysis['distances']
        linear_line = analysis['linear_line']
        kneedle_idx = analysis['kneedle_idx']
        r_kneedle = analysis['r_kneedle']
        r_star = analysis['r_star']
        spline = analysis['spline']
        inflection_candidates = analysis['inflection_candidates']
        
        x_raw = np.array(ranks)
        
        plt.figure(figsize=(10, 6))
        
        # Normalized error
        plt.plot(x, y, 'o', color='orange', label='Normalized Error $y_i$')
        
        # Spline fit
        if spline is not None:
            x_dense = np.linspace(0, 1, 500)
            y_spline = spline(x_dense)
            plt.plot(x_dense, y_spline, '-', color='crimson', label='Spline Fit $S(x)$')
        
        # Reference line
        plt.plot(x, linear_line, '--', color='orangered', label='Reference Line $1 - x$')
        
        # Absolute distances
        plt.plot(x, distances, 'D--', color='blue', alpha=0.7, label='|Distance from $1 - x$|')
        
        # Inflection points
        if len(inflection_candidates) > 0 and spline is not None:
            y_inflect = spline(inflection_candidates)
            plt.scatter(inflection_candidates, y_inflect, color='green', zorder=5, label='Inflection Point(s)')
        
        # Final selection markers
        r_kneedle_norm = x[kneedle_idx]
        plt.axvline(r_kneedle_norm, color='green', linestyle='--', label=f'Kneedle: r={r_kneedle}')
        
        # Selected rank line
        plt.axvline(np.interp(r_star, x_raw, x), color='black', linestyle='--', label=f'Selected Rank: r={r_star}')
        
        # Final styling
        plt.xlabel('Normalized Rank $x$')
        plt.ylabel('Value (Error or Distance)')
        plt.title('Kneedle Curve with Absolute Distance and Inflection Points')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Rank selection plot saved to {plot_path}")