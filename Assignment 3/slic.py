import numpy as np
from numpy.linalg import norm
from skimage.color import rgb2lab


def my_slic(image: np.ndarray, compactness: int, num_clusters: int,
            num_iterations: int = 10, connected_components: bool = True) -> np.ndarray:
    """Implements an SLIC algorithm as described in the original paper and commonly implemented

    Args:
        image: The input image in RGB space, of size MxNx3
        compactness: The compactness parameter, higher values weight spatial distance more
        num_clusters: The approximate number of clusters to form
        num_iterations: The number of iterations to run
        connected_components: Whether to enforce connectivity of the segments

    Returns:
        An MxN array, where each pixel is assigned to a cluster
    """

    # Convert to CIELAB space
    m = image.shape[0]
    n = image.shape[1]
    image_lab = rgb2lab(image)
    image_xy = np.argwhere(np.ones(image_lab.shape[:2])).reshape(m, n, 2)
    image_labxy = np.concatenate([image_lab, image_xy], axis=2)

    # Initialize cluster centers
    grid_interval = int(np.sqrt(m * n / num_clusters))
    center_positions = np.array([[i, j] for i in range(grid_interval // 2, m, grid_interval)
                                 for j in range(grid_interval // 2, n, grid_interval)])

    # Move each center the pixel with the least change in color in a 3x3 window (the minimum gradient), in order to
    # avoid placing centers on edges
    # Using 2 pixels of padding on each side to deal with edge cases
    padded_lab = np.pad(image_lab, ((2, 2), (2, 2), (0, 0)), constant_values=np.mean(image_lab))
    for idx, center in enumerate(center_positions + (2, 2)):
        i, j = center
        min_gradient = np.inf
        best_position = (i, j)
        for x in range(i - 1, i + 2):
            for y in range(j - 1, j + 2):
                gradient = (norm(padded_lab[x + 1, y] - padded_lab[x - 1, y]) +
                            norm(padded_lab[x, y + 1] - padded_lab[x, y - 1]))
                if gradient < min_gradient:
                    min_gradient = gradient
                    best_position = (x, y)
        center_positions[idx] = best_position

    # Get the finalized initial cluster centers
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
            d = np.sqrt(d_color + (d_space / grid_interval ** 2) * (compactness ** 2))

            # Update the pixels who are closer to this cluster center
            update_indices = np.where(d < distances[xx, yy])
            distances[xx[update_indices], yy[update_indices]] = d[update_indices]
            segments[xx[update_indices], yy[update_indices]] = idx

        # Update the location of the cluster centers
        for idx in range(len(centers)):
            segment_mask = segments == idx
            segment_labxy = image_labxy[segment_mask]
            segment_mean = np.mean(segment_labxy, axis=0)
            centers[idx] = segment_mean

    # Since SLIC can produce many disconnected segments, a separate algorithm is used to enforce connectivity
    if connected_components:
        segments = enforce_connectivity(segments, num_clusters)

    return segments


def enforce_connectivity(segments: np.ndarray, num_clusters: int) -> np.ndarray:
    """Enforces connectivity of the segments, following a common algorithm found in SLIC implementations

    Args:
        segments: The segments to enforce connectivity on
        num_clusters: The number of clusters to enforce connectivity for

    Returns:
        An MxN array, where each pixel is assigned to a cluster index, with connectivity enforced

    """

    m = segments.shape[0]
    n = segments.shape[1]
    neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    min_segment_size = (m * n) // num_clusters / 4  # This value could probably be tuned

    new_clusters = np.full_like(segments, fill_value=-1, dtype=int)

    segment_label = 0
    for i in range(m):
        for j in range(n):
            if new_clusters[i, j] > -1:
                continue

            # BFS to find all connected pixels in the same segment (like flood fill)
            connected_segment = []
            next_pixels = [(i, j)]
            while len(next_pixels):
                px, py = next_pixels.pop(0)
                connected_segment.append((px, py))
                for dx, dy in neighbors:
                    x, y = px + dx, py + dy
                    if 0 <= x < m and 0 <= y < n and new_clusters[x, y] == -1 and segments[i, j] == segments[x, y]:
                        next_pixels.append((x, y))
                        new_clusters[x, y] = segment_label

            # If the segment is too small, merge it with an adjacent segment
            if len(connected_segment) < min_segment_size:
                # Find an adjacent segment
                adj_label = segment_label + 1  # should be overridden
                for dx, dy in neighbors:
                    x, y = i + dx, j + dy
                    if 0 <= x < m and 0 <= y < n and new_clusters[x, y] >= 0 and new_clusters[x, y] != segment_label:
                        adj_label = new_clusters[x, y]

                # Reassign the pixels
                for x, y in connected_segment:
                    new_clusters[x, y] = adj_label
            else:
                # Increment the current segment label if the segment wasn't merged
                segment_label += 1

    return new_clusters


def segments_to_image(image: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Uses an array of segments to cluster an image into superpixels.

    Args:
        image: The input image, of size MxNx3
        segments: The segments to cluster the image into, of size MxN

    Returns:
        An MxNx3 image, where each pixel is set to the mean color of the corresponding segment

    """

    segmented_image = np.zeros_like(image)
    for segment_label in np.unique(segments):
        segment_mask = segments == segment_label
        segment_mean_color = np.mean(image[segment_mask], axis=0)
        segmented_image[segment_mask] = segment_mean_color
    return segmented_image
