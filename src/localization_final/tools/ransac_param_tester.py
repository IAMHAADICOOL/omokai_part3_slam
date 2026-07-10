#!/usr/bin/env python3
import time
import math
import numpy as np

class PolarLineExtractorCore:
    def __init__(self, min_points, max_dist, max_iter, min_len):
        self.min_points_per_line = min_points
        self.max_line_distance = max_dist
        self.max_iterations = max_iter
        self.min_line_length = min_len
        self.rng = np.random.default_rng()

    def extract_lines(self, points):
        if points.shape[0] < self.min_points_per_line:
            return [], []

        remaining = points.copy()
        lines = []
        errors = []

        while remaining.shape[0] >= self.min_points_per_line:
            best = self._ransac_best_line(remaining)
            if best is None:
                break

            best_inlier_mask, best_count = best
            if best_count < self.min_points_per_line:
                break

            inliers = remaining[best_inlier_mask]
            refit = self._refit_line_svd(inliers)

            if refit is None:
                remaining = remaining[~best_inlier_mask]
                continue

            line_normal, line_direction, line_point = refit
            p0, p1, line_length = self._line_segment_from_inliers(inliers, line_direction, line_point)

            if line_length >= self.min_line_length:
                dists = self._point_to_line_distances(inliers, p0, p1)
                mean_err = np.mean(dists)
                
                lines.append((p0, p1))
                errors.append(mean_err)

            remaining = remaining[~best_inlier_mask]

        return lines, errors

    def _ransac_best_line(self, points):
        n = points.shape[0]
        if n < 2: return None
        best_count = 0
        best_inlier_mask = None

        for _ in range(self.max_iterations):
            idx = self.rng.choice(n, size=2, replace=False)
            p1, p2 = points[idx[0]], points[idx[1]]
            if np.linalg.norm(p2 - p1) < 1e-9: continue

            dists = self._point_to_line_distances(points, p1, p2)
            inlier_mask = dists <= self.max_line_distance
            count = int(np.count_nonzero(inlier_mask))

            if count > best_count:
                best_count = count
                best_inlier_mask = inlier_mask

        return (best_inlier_mask, best_count) if best_inlier_mask is not None else None

    @staticmethod
    def _point_to_line_distances(points, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        numerator = np.abs((y2 - y1)*points[:,0] - (x2 - x1)*points[:,1] + x2*y1 - y2*x1)
        den = math.hypot(y2 - y1, x2 - x1)
        if den < 1e-12: return np.full(points.shape[0], np.inf)
        return numerator / den

    @staticmethod
    def _refit_line_svd(inliers):
        if inliers.shape[0] < 2: return None
        centroid = np.mean(inliers, axis=0)
        centered = inliers - centroid
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        direction = vh[0]
        if np.linalg.norm(direction) < 1e-12: return None
        direction = direction / np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        if np.linalg.norm(normal) < 1e-12: return None
        return normal / np.linalg.norm(normal), direction, centroid

    @staticmethod
    def _line_segment_from_inliers(inliers, direction, point):
        t = (inliers - point) @ direction
        t_min, t_max = float(np.min(t)), float(np.max(t))
        p0 = point + t_min * direction
        p1 = point + t_max * direction
        return p0, p1, float(np.linalg.norm(p1 - p0))


def generate_synthetic_room(n_points=400, noise=0.015):
    # 4 walls of a 4x4m room centered at 0,0
    pts = []
    per_wall = n_points // 4
    rng = np.random.default_rng(42)
    
    # Top wall (y=2)
    x = rng.uniform(-2, 2, per_wall)
    y = np.full(per_wall, 2.0) + rng.normal(0, noise, per_wall)
    pts.append(np.column_stack((x, y)))
    
    # Bottom wall (y=-2)
    x = rng.uniform(-2, 2, per_wall)
    y = np.full(per_wall, -2.0) + rng.normal(0, noise, per_wall)
    pts.append(np.column_stack((x, y)))
    
    # Left wall (x=-2)
    y = rng.uniform(-2, 2, per_wall)
    x = np.full(per_wall, -2.0) + rng.normal(0, noise, per_wall)
    pts.append(np.column_stack((x, y)))
    
    # Right wall (x=2)
    y = rng.uniform(-2, 2, per_wall)
    x = np.full(per_wall, 2.0) + rng.normal(0, noise, per_wall)
    pts.append(np.column_stack((x, y)))
    
    return np.vstack(pts)


def main():
    print("--- RANSAC Parameter Tuning Test ---")
    points = generate_synthetic_room(n_points=600, noise=0.02)
    print(f"Generated synthetic room with {points.shape[0]} points and 0.02m noise.\n")

    max_iter_list = [50, 100, 200, 500]
    max_dist_list = [0.03, 0.05, 0.08, 0.10]
    
    print(f"{'max_iter':<10} | {'max_dist':<10} | {'Lines':<7} | {'Mean Err (m)':<13} | {'Time (ms)':<10}")
    print("-" * 60)

    for mi in max_iter_list:
        for md in max_dist_list:
            extractor = PolarLineExtractorCore(min_points=20, max_dist=md, max_iter=mi, min_len=0.3)
            
            total_time = 0
            for _ in range(5):
                t0 = time.perf_counter()
                lines, errors = extractor.extract_lines(points)
                total_time += (time.perf_counter() - t0) * 1000.0
                
            avg_time = total_time / 5.0
            avg_err = np.mean(errors) if errors else 0.0
            
            print(f"{mi:<10} | {md:<10} | {len(lines):<7} | {avg_err:<13.4f} | {avg_time:<10.2f}")

if __name__ == '__main__':
    main()
