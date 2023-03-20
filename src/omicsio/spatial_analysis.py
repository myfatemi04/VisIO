from .datasets import Slide

from typing import List, Optional

import numpy as np
import pandas as pd
import torch

def run_spatialde(counts: np.ndarray, genes: List[str], image_x: np.ndarray, image_y: np.ndarray):
    import SpatialDE
    import NaiveDE

    assert np.sum(counts < 0) == 0, "Counts must be non-negative"

    sample = pd.DataFrame({
        'total_counts': counts.sum(axis=1),
        'image_x': image_x,
        'image_y': image_y,
    })

    counts_df = pd.DataFrame(counts, columns=genes)

    normalized_counts = NaiveDE.stabilize(counts_df.T).T
    residual_expression = NaiveDE.regress_out(sample, normalized_counts.T, 'np.log(total_counts)').T

    sample_resid_exp = residual_expression.sample(n=len(genes), axis=1, random_state=1)
    results = SpatialDE.run(sample[['image_x', 'image_y']], sample_resid_exp)
    
    return results

def run_aeh(
    counts: np.ndarray,
    genes: List[str],
    image_x: np.ndarray,
    image_y: np.ndarray,
    num_groups: int,
    spatialde_results: pd.DataFrame,
    length: float,
    significance_filter: str = 'qval < 0.05'
):
    import SpatialDE
    import NaiveDE

    sample = pd.DataFrame({
        'total_counts': counts.sum(axis=1),
        'image_x': image_x,
        'image_y': image_y,
    })

    counts_df = pd.DataFrame(counts, columns=genes)

    normalized_counts = NaiveDE.stabilize(counts_df.T).T
    residual_expression = NaiveDE.regress_out(sample, normalized_counts.T, 'np.log(total_counts)').T

    significant_results = spatialde_results.query(significance_filter)

    histology_results, patterns = SpatialDE.aeh.spatial_patterns(sample[['image_x', 'image_y']], residual_expression, significant_results, C=num_groups, l=length, verbosity=1)
    
    return histology_results, patterns

def run_spatialde_and_aeh_on_slide(slide: Slide, num_groups: int, custom_spot_counts: Optional[torch.Tensor] = None):
    if custom_spot_counts is not None:
        spot_counts = custom_spot_counts
    else:
        spot_counts = slide.spot_counts

    spatialde_result = run_spatialde(
        spot_counts.cpu().numpy(),
        slide.genes,
        slide.spot_locations.image_x.numpy(),
        slide.spot_locations.image_y.numpy(),
    )

    aeh_histology_results, aeh_patterns = run_aeh(
        spot_counts.cpu().numpy(),
        slide.genes,
        slide.spot_locations.image_x.numpy(),
        slide.spot_locations.image_y.numpy(),
        num_groups=num_groups,
        spatialde_results=spatialde_result,
        length=0.1,
        significance_filter='qval < 0.05'
    )

    return {"spatialde_result": spatialde_result, "aeh_histology_results": aeh_histology_results, "aeh_patterns": aeh_patterns}
