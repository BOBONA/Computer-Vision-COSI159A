from pathlib import Path

from skimage.io import imread, imsave
from skimage.segmentation import slic

import args
from slic import my_slic, segments_to_image

if __name__ == '__main__':
    image_path = Path(args.image)
    output_path = args.output if args.output \
        else image_path.parent / f"segmented_{args.algorithm}_{args.clusters}_{args.compactness}_{image_path.name}.png"
    image = imread(image_path)

    segments = slic(image, n_segments=args.clusters, max_num_iter=args.iterations,
                    enforce_connectivity=args.connected, start_label=0) if args.algorithm == 'scikit' \
        else my_slic(image, compactness=args.compactness, num_clusters=args.clusters,
                     num_iterations=args.iterations, connected_components=args.connected)

    segmented_image = segments_to_image(image, segments)
    imsave(output_path, segmented_image)
