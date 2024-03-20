import argparse

parser = argparse.ArgumentParser(prog="Simpler Linear Iterative Clustering")

parser.add_argument('--image', type=str, help="Path to the input image", default="data/example1.jpg")
parser.add_argument('--output', type=str, help="Path to save the output image", default="")
parser.add_argument('--algorithm', type=str, help="Algorithm to use; can be 'scikit' or 'my-slic'", default='my-slic')
parser.add_argument('--clusters', type=int, help="Number of clusters to form", default=100)
parser.add_argument('--compactness', type=int, help="SLIC compactness parameter", default=10)
parser.add_argument('--iterations', type=int, help="Number of iterations to run", default=10)
args = parser.parse_args()

image: str = args.image
output: str = args.output
algorithm: str = args.algorithm
clusters: int = args.clusters
compactness: int = args.compactness
iterations: int = args.iterations
