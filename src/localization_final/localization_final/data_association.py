import numpy as np
import scipy.stats
from math import cos, sin

from .utils import warp_angle


class DataAssociation:
    def __init__(
        self,
        confidence_level,
        map_feature,
        map_feature_cov,
        allow_landmark_reuse=True,
        duplicate_rho_threshold=0.15,
        duplicate_alpha_threshold=0.15,
        measurement_sigma_rho_floor=0.04,
        measurement_sigma_alpha_floor=0.06,
    ):
        self.confidence_level = float(confidence_level)
        self.map_feature = map_feature
        self.map_feature_cov = map_feature_cov
        self.nf = len(map_feature)
        self.zfi_dim = 2

        self.allow_landmark_reuse = bool(allow_landmark_reuse)

        self.duplicate_rho_threshold = float(duplicate_rho_threshold)
        self.duplicate_alpha_threshold = float(duplicate_alpha_threshold)

        self.measurement_sigma_rho_floor = float(measurement_sigma_rho_floor)
        self.measurement_sigma_alpha_floor = float(measurement_sigma_alpha_floor)

        self.last_distance_matrix = None
        self.last_compatible_matrix = None
        self.raw_duplicate_count = 0
        self.recovered_duplicates = 0
        self.geometry_rescued_count = 0

    def _safe_covariance(self, cov, dim=2, apply_polar_floor=True):
        # Symmetrise and regularise to ensure positive-definiteness before matrix inversion
        cov = np.asarray(cov, dtype=float)

        if cov.shape != (dim, dim):
            cov = np.eye(dim) * 0.05

        cov = 0.5 * (cov + cov.T)
        cov += np.eye(dim) * 1e-9

        # Measurement uncertainty floors prevent over-confident polar line associations
        if dim == 2 and apply_polar_floor:
            cov[0, 0] = max(cov[0, 0], self.measurement_sigma_rho_floor**2)
            cov[1, 1] = max(cov[1, 1], self.measurement_sigma_alpha_floor**2)

        return cov

    def _normalize_predicted_observation(self, rho, alpha):
        flipped = False

        if rho < 0.0:
            rho = -rho
            alpha = warp_angle(alpha + np.pi)
            flipped = True
        else:
            alpha = warp_angle(alpha)

        return float(rho), float(alpha), flipped

    def _normalize_world_feature(self, rho, alpha):
        rho = float(rho)
        alpha = float(alpha)

        if rho < 0.0:
            rho = -rho
            alpha = warp_angle(alpha + np.pi)
        else:
            alpha = warp_angle(alpha)

        return [float(rho), float(alpha)]

    def hfj(self, xk, j):
        # Predicted observation h(X_k, f_j): projects world-frame polar landmark L_j = (ρ_w, α_w)
        # into the robot frame at pose X_k = (x, y, θ):
        #   ρ̂ = ρ_w − x cos α_w − y sin α_w
        #   α̂ = wrap(α_w − θ)
        rho_w = float(self.map_feature[j][0])
        alpha_w = float(self.map_feature[j][1])

        x_robot = float(xk[0, 0])
        y_robot = float(xk[1, 0])
        theta_robot = float(xk[2, 0])

        rho_obs = rho_w - cos(alpha_w) * x_robot - sin(alpha_w) * y_robot
        alpha_obs = alpha_w - theta_robot

        rho_obs, alpha_obs, _ = self._normalize_predicted_observation(
            rho_obs,
            alpha_obs,
        )

        return np.array([[rho_obs], [alpha_obs]], dtype=float)

    def Jhfjx(self, xk, j):
        # H_x = ∂h/∂X_k — measurement Jacobian with respect to robot pose (2×3 matrix):
        #   row 0: [−cos α_w, −sin α_w, 0]
        #   row 1: [0, 0, −1]
        # First row changes sign if normalisation required ρ̂ flip
        rho_w = float(self.map_feature[j][0])
        alpha_w = float(self.map_feature[j][1])

        x_robot = float(xk[0, 0])
        y_robot = float(xk[1, 0])
        theta_robot = float(xk[2, 0])

        rho_obs = rho_w - cos(alpha_w) * x_robot - sin(alpha_w) * y_robot
        _, _, flipped = self._normalize_predicted_observation(
            rho_obs,
            alpha_w - theta_robot,
        )

        Hx = np.array(
            [
                [-cos(alpha_w), -sin(alpha_w), 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        )

        if flipped:
            Hx[0, :] *= -1.0

        return Hx

    def Jhfjf(self, xk, j):
        # H_f = ∂h/∂L_j — measurement Jacobian with respect to landmark parameters (ρ_w, α_w) (2×2 matrix):
        #   [[1,  x sin α_w − y cos α_w],
        #    [0,  1                     ]]
        # Upper row changes sign if normalisation required ρ̂ flip
        rho_w = float(self.map_feature[j][0])
        alpha_w = float(self.map_feature[j][1])

        x_robot = float(xk[0, 0])
        y_robot = float(xk[1, 0])
        theta_robot = float(xk[2, 0])

        rho_obs = rho_w - cos(alpha_w) * x_robot - sin(alpha_w) * y_robot
        _, _, flipped = self._normalize_predicted_observation(
            rho_obs,
            alpha_w - theta_robot,
        )

        Hf = np.array(
            [
                [1.0, x_robot * sin(alpha_w) - y_robot * cos(alpha_w)],
                [0.0, 1.0],
            ],
            dtype=float,
        )

        if flipped:
            Hf[0, :] *= -1.0

        return Hf

    def expected_observation_covariance(self, xk, Pk, j):
        # Predicted observation covariance via first-order error propagation:
        #   P_ẑ = H_x P_k H_x^T + H_f P_f H_f^T
        # This quantifies uncertainty in the predicted observation due to robot pose and landmark covariances
        Hx = self.Jhfjx(xk, j)
        Hf = self.Jhfjf(xk, j)

        Pk = self._safe_covariance(Pk, dim=3, apply_polar_floor=False)

        if j < len(self.map_feature_cov):
            Pf = self._safe_covariance(
                self.map_feature_cov[j],
                dim=2,
                apply_polar_floor=True,
            )
        else:
            Pf = np.diag(
                [
                    self.measurement_sigma_rho_floor**2,
                    self.measurement_sigma_alpha_floor**2,
                ]
            )

        Ph = Hx @ Pk @ Hx.T + Hf @ Pf @ Hf.T
        Ph = self._safe_covariance(Ph, dim=2, apply_polar_floor=True)

        return Ph

    def SquaredMahalanobisDistance(self, hfj, Pfj, zfi, Rfi):
        # D²_ij = ν^T S^{-1} ν  where ν = z_i − ĥ_j (innovation) and S = P_ẑ + R_i (total innovation covariance)
        # Angular innovation component is wrapped to (-π, π] to avoid discontinuity at ±π
        hfj = np.asarray(hfj, dtype=float).reshape(2, 1)
        zfi = np.asarray(zfi, dtype=float).reshape(2, 1)

        innovation = zfi - hfj
        innovation[1, 0] = warp_angle(innovation[1, 0])

        S = (
            self._safe_covariance(Pfj, dim=2, apply_polar_floor=True)
            + self._safe_covariance(Rfi, dim=2, apply_polar_floor=True)
        )

        S = self._safe_covariance(S, dim=2, apply_polar_floor=True)

        try:
            D2 = float(innovation.T @ np.linalg.inv(S) @ innovation)
        except np.linalg.LinAlgError:
            # Moore-Penrose pseudo-inverse as fallback when S is numerically singular
            D2 = float(innovation.T @ np.linalg.pinv(S) @ innovation)

        return D2

    def IndividualCompatibility(self, D2_ij, dof, alpha):
        # χ²(α, dof=2) gate: retains pairs (i, j) whose squared Mahalanobis distance lies within the confidence ellipsoid
        threshold = scipy.stats.chi2.ppf(alpha, dof)
        return D2_ij <= threshold

    def _feature_geometry_score(self, candidate, feature):
        rho_c = float(candidate[0])
        alpha_c = float(candidate[1])

        rho_f = float(feature[0])
        alpha_f = float(feature[1])

        rho_error = abs(rho_c - rho_f)
        alpha_error = abs(warp_angle(alpha_c - alpha_f))

        if rho_error > self.duplicate_rho_threshold:
            return None

        if alpha_error > self.duplicate_alpha_threshold:
            return None

        score = (
            rho_error / max(self.duplicate_rho_threshold, 1e-9)
            + alpha_error / max(self.duplicate_alpha_threshold, 1e-9)
        )

        return float(score)

    def find_geometry_duplicate(self, candidate_feature):
        if len(self.map_feature) == 0:
            return None

        best_j = None
        best_score = np.inf

        for j, feature in enumerate(self.map_feature):
            score = self._feature_geometry_score(candidate_feature, feature)

            if score is None:
                continue

            if score < best_score:
                best_score = score
                best_j = j

        return best_j

    def observation_to_world_feature(self, xk, Pk, zfi, Rfi):
        x_robot = float(xk[0, 0])
        y_robot = float(xk[1, 0])
        theta_robot = float(xk[2, 0])

        rho_obs = float(zfi[0])
        alpha_obs = float(zfi[1])

        alpha_w = warp_angle(alpha_obs + theta_robot)
        rho_w = rho_obs + cos(alpha_w) * x_robot + sin(alpha_w) * y_robot

        rho_w, alpha_w = self._normalize_world_feature(rho_w, alpha_w)

        Jx = np.array(
            [
                [
                    cos(alpha_w),
                    sin(alpha_w),
                    -x_robot * sin(alpha_w) + y_robot * cos(alpha_w),
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        Jz = np.array(
            [
                [
                    1.0,
                    -x_robot * sin(alpha_w) + y_robot * cos(alpha_w),
                ],
                [0.0, 1.0],
            ],
            dtype=float,
        )

        Pk = self._safe_covariance(Pk, dim=3, apply_polar_floor=False)
        Rfi = self._safe_covariance(Rfi, dim=2, apply_polar_floor=True)

        cov_feature_map = Jx @ Pk @ Jx.T + Jz @ Rfi @ Jz.T
        cov_feature_map = self._safe_covariance(
            cov_feature_map,
            dim=2,
            apply_polar_floor=True,
        )

        return [float(rho_w), float(alpha_w)], cov_feature_map

    def ICNN(self, hf, Phf, zf, Rf, dim):
        n_obs = len(zf)
        n_map = len(hf)

        H = [None for _ in range(n_obs)]

        if n_obs == 0 or n_map == 0:
            self.last_distance_matrix = np.empty((n_obs, n_map))
            self.last_compatible_matrix = np.zeros((n_obs, n_map), dtype=bool)
            self.raw_duplicate_count = 0
            self.recovered_duplicates = 0
            return H

        D = np.full((n_obs, n_map), np.inf, dtype=float)
        C = np.zeros((n_obs, n_map), dtype=bool)

        raw_nearest = []

        for i in range(n_obs):
            best_j = None
            best_d = np.inf

            for j in range(n_map):
                D2_ij = self.SquaredMahalanobisDistance(
                    hf[j],
                    Phf[j],
                    zf[i],
                    Rf[i],
                )

                D[i, j] = D2_ij

                if self.IndividualCompatibility(D2_ij, dim, self.confidence_level):
                    C[i, j] = True

                    if D2_ij < best_d:
                        best_d = D2_ij
                        best_j = j

            raw_nearest.append(best_j)

        used_raw = {}

        for j in raw_nearest:
            if j is None:
                continue

            used_raw[j] = used_raw.get(j, 0) + 1

        duplicate_count = 0

        for _, count in used_raw.items():
            if count > 1:
                duplicate_count += count - 1

        self.raw_duplicate_count = duplicate_count

        compatible_pairs = []

        for i in range(n_obs):
            for j in range(n_map):
                if C[i, j]:
                    compatible_pairs.append((D[i, j], i, j))

        compatible_pairs.sort(key=lambda item: item[0])

        assigned_observations = set()
        assigned_landmarks = set()

        for _, i, j in compatible_pairs:
            if i in assigned_observations:
                continue

            if not self.allow_landmark_reuse and j in assigned_landmarks:
                continue

            H[i] = j
            assigned_observations.add(i)
            assigned_landmarks.add(j)

        self.last_distance_matrix = D
        self.last_compatible_matrix = C

        return H

    def DataAssociation(self, xk, Pk, zf, Rf):
        self.nf = len(self.map_feature)

        h_F = []
        P_F = []

        for j in range(self.nf):
            h_Fj = self.hfj(xk, j)
            P_Fj = self.expected_observation_covariance(xk, Pk, j)

            h_F.append(h_Fj)
            P_F.append(P_Fj)

        H = self.ICNN(h_F, P_F, zf, Rf, self.zfi_dim)

        geometry_rescued = 0

        for i in range(len(zf)):
            if H[i] is not None:
                continue

            candidate_feature, _ = self.observation_to_world_feature(
                xk,
                Pk,
                zf[i],
                Rf[i],
            )

            duplicate_id = self.find_geometry_duplicate(candidate_feature)

            if duplicate_id is not None:
                H[i] = duplicate_id
                geometry_rescued += 1

        self.geometry_rescued_count = geometry_rescued
        self.recovered_duplicates = self.raw_duplicate_count + geometry_rescued

        return H

    def AddNewFeature(self, xk, Pk, zfi, Rfi):
        feature, cov_feature_map = self.observation_to_world_feature(
            xk,
            Pk,
            zfi,
            Rfi,
        )

        duplicate_id = self.find_geometry_duplicate(feature)

        if duplicate_id is not None:
            return duplicate_id, False

        self.map_feature.append(feature)
        self.map_feature_cov.append(cov_feature_map)

        return len(self.map_feature) - 1, True

    def AddmultipleNewFeatures(self, xk, Pk, zf, Rf):
        for i in range(len(zf)):
            self.AddNewFeature(xk, Pk, zf[i], Rf[i])

        return self.map_feature, self.map_feature_cov

    def GetUnassociatedFeatures(self, zf, Rf, H):
        unassociated_features = []
        unassociated_features_cov = []

        for i in range(len(zf)):
            if H[i] is None:
                unassociated_features.append(zf[i])
                unassociated_features_cov.append(Rf[i])

        return unassociated_features, unassociated_features_cov