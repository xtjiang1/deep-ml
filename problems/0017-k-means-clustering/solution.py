import numpy as np
def eucliden_distance(point: tuple[float, ...], initial_centroid: tuple[float, ...]) -> float:
	distance = np.linalg.norm(point - initial_centroid)
	return distance



def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	points = np.array(points, dtype=float)
	initial_centroids = np.array(initial_centroids, dtype=float)
	distances = np.zeros((points.shape[0], k))

	max_distances_index = np.zeros(points.shape[0])

	for _ in range(max_iterations):
		new_centroids = np.zeros_like(initial_centroids)
		for i, point in enumerate(points):
			for j, c in enumerate(initial_centroids):
				distances[i, j] = eucliden_distance(point, c)
			max_distances_index[i] = np.argmin(distances[i], axis=0)
		unique_labels = np.unique(max_distances_index)
		for  labels in  unique_labels:
			group_points = points[max_distances_index == labels]
			new_centroids[int(labels)] = np.mean(group_points, axis=0)
		initial_centroids = new_centroids.copy()
	
	return [tuple(centroid) for centroid in np.round(new_centroids, decimals=4).tolist()]