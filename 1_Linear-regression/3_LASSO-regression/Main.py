#
# Copyright (c) 2016-2021 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from sklearn.utils import arrayfuncs
from sklearn import datasets


class LarsLasso:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def predict(self, X: np.ndarray):
        y = np.dot(X, self.coef_) + self.intercept_
        return y

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, p = X.shape

        self.intercept_ = y.mean()
        y = y - self.intercept_
        self.coef_ = np.zeros(p)

        active_set = []
        inactive_set = list(range(p))
        beta = np.zeros(p)
        mu = np.zeros(n)

        change_sign_flag = False

        k = 0
        while k != min(p, n - 1):
            c = np.dot(X.T, y - mu)
            # print(np.sign(c[active_set]) * np.sign(beta[active_set]))
            if change_sign_flag:
                # remove j from the calculation of the next equiangular direction.
                pass
            else:
                j = inactive_set[np.argmax(np.abs(c[inactive_set]))]
                # print(f"add {j=}")
                active_set.append(j)
                inactive_set.remove(j)
            C = np.amax(np.abs(c))
            s = np.sign(c[active_set]).reshape((1, len(active_set)))
            XA = np.copy(X[:, active_set] * s)

            GA = XA.T @ XA
            GA_inv = np.linalg.inv(GA)

            one = np.ones((len(active_set), 1))
            AA = (1. / np.sqrt(one.T @ GA_inv @ one)).flatten()[0]

            w = AA * GA_inv @ one
            u = XA @ w

            a = X.T @ u
            d = s.T * w

            if k == p - 1:
                gamma = C / AA
            else:
                gamma_candidates = np.zeros((len(inactive_set), 2))
                for _j, jj in enumerate(inactive_set):
                    gamma_candidates[_j] = [(C - c[jj]) / (AA - a[jj]), (C + c[jj]) / (AA + a[jj])]
                gamma = arrayfuncs.min_pos(gamma_candidates)

            gamma_candidates_tilde = - beta[active_set] / d.flatten()
            gamma_tilde = arrayfuncs.min_pos(gamma_candidates_tilde)

            change_sign_flag = False
            if gamma_tilde < gamma:
                gamma = gamma_tilde
                j = active_set[list(gamma_candidates_tilde).index(gamma)]
                # print(f"remove {j=}")
                change_sign_flag = True

            new_beta = beta[active_set] + gamma * d.flatten()
            idx = 0 if j != 0 else 1
            tmp_beta = np.zeros(p)
            tmp_beta[active_set] = new_beta.copy()
            lambda_ = np.abs(X[:, active_set[idx]] @ (y - X @ tmp_beta)) * 2 / n

            if lambda_ < self.alpha:
                prev_lambda_ = np.abs(X[:, active_set[idx]] @ (y - X @ self.coef_)) * 2 / n
                if len(active_set) < 2 and prev_lambda_ < self.alpha:
                    break
                # print(prev_lambda_, lambda_, self.alpha)
                modified_gamma = 0 + (gamma - 0) * (self.alpha - prev_lambda_) / (lambda_ - prev_lambda_)
                beta[active_set] += modified_gamma * d.flatten()
                mu = mu + modified_gamma * u.flatten()
                self.coef_ = beta.copy()
                # print(np.abs(X[:, active_set[idx]] @ (y - X @ self.coef_)) * 2 / n)
                break

            mu = mu + gamma * u.flatten()
            beta[active_set] = new_beta.copy()
            self.coef_ = beta.copy()
            # print(self.coef_)

            if change_sign_flag:
                active_set.remove(j)
                inactive_set.append(j)
            k = len(active_set)
        return self


if __name__ == "__main__":
    dataset = datasets.load_boston()
    X = dataset.data
    y = dataset.target

    X = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
    model = LarsLasso(alpha=1)
    model.fit(X, y)

    print(model.intercept_)
    print(model.coef_)
