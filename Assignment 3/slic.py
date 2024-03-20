import numpy as np
from skimage.color import rgb2lab


def my_slic(image_lab: np.ndarray, compactness: int, num_clusters: int, num_iterations: int = 10) -> np.ndarray:
    """Implements a standard SLIC algorithm.

    Args:
        image_lab: The input image in RGB space, of size MxNx3.
        compactness: The compactness parameter, higher values weight spatial distance more.
        num_clusters: The approximate number of clusters to form.
        num_iterations: The number of iterations to run.

    Returns:
        An MxN array, where each pixel is assigned to a cluster.
    """

    # Convert to CIELAB space
    image_lab = rgb2lab(image_lab)
    m = image_lab.shape[0]
    n = image_lab.shape[1]
    image_xy = np.argwhere(np.ones(image_lab.shape[:2])).reshape(m, n, 2)
    image_labxy = np.concatenate([image_lab, image_xy], axis=2)

    # Initialize cluster centers
    grid_interval = int(np.sqrt(m * n / num_clusters))
    center_positions = np.array([[i, j] for i in range(grid_interval // 2, m, grid_interval)
                                 for j in range(grid_interval // 2, n, grid_interval)])
    centers = np.array([[*image_lab[x, y], x, y] for x, y in center_positions])

    # Initialize segments
    segments = np.zeros(image_lab.shape[:2], dtype=int)
    distances = np.full(image_lab.shape[:2], np.inf)

    # Iterate
    for _ in range(num_iterations):
        # Assign pixels to clusters
        for idx, center in enumerate(centers):
            cl, ca, cb, cx, cy = center
            cx, cy = int(cx), int(cy)

            # Get the window around the cluster center
            x_range = range(max(0, cx - grid_interval), min(m, cx + grid_interval))
            y_range = range(max(0, cy - grid_interval), min(n, cy + grid_interval))
            xx, yy = np.meshgrid(x_range, y_range)
            center_window = image_lab[xx, yy]

            # Vectorized distance calculations
            d_color = np.sum((center_window - [cl, ca, cb]) ** 2, axis=2)
            d_space = (xx - cx) ** 2 + (yy - cy) ** 2
            d = np.sqrt((d_color ** 2) + (d_space ** 2 / grid_interval ** 2) * (compactness ** 2))

            update_indices = np.where(d < distances[xx, yy])
            distances[xx[update_indices], yy[update_indices]] = d[update_indices]
            segments[xx[update_indices], yy[update_indices]] = idx

            # # Iterate over pixels in a window around the cluster center
            # for x in range(max(0, cx - grid_interval), min(m, cx + grid_interval)):
            #     for y in range(max(0, cy - grid_interval), min(n, cy + grid_interval)):
            #         l, a, b = image_lab[x, y]
            #
            #         # Calculate the distance, weighted by the compactness parameter
            #         d_color = (l - cl) ** 2 + (a - ca) ** 2 + (b - cb) ** 2
            #         d_space = (x - cx) ** 2 + (y - cy) ** 2
            #         d = np.sqrt((d_color ** 2) + (d_space ** 2 / grid_interval ** 2) * (compactness ** 2))
            #
            #         # Update the assigned cluster if necessary
            #         if d < distances[x, y]:
            #             distances[x, y] = d
            #             segments[x, y] = idx

        # Update the cluster centers
        for idx in range(len(centers)):
            segment_mask = segments == idx
            segment_labxy = image_labxy[segment_mask]
            segment_mean = np.mean(segment_labxy, axis=0)
            centers[idx] = segment_mean

    return segments


def segments_to_image(image: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Uses an array of segments to cluster an image into superpixels."""
    segmented_image = np.zeros_like(image)
    for segment_label in np.unique(segments):
        segment_mask = segments == segment_label
        segment_mean_color = np.mean(image[segment_mask], axis=0)
        segmented_image[segment_mask] = segment_mean_color
    return segmented_image
