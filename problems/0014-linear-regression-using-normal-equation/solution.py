import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	theta = np.linalg.inv(np.array(X).T @ np.array(X)) @ np.array(X).T @ np.array(y)
	return theta