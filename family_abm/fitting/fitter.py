from __future__ import annotations
from typing import Any, Callable, Optional
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
from .lanchester import MODEL_REGISTRY, MODEL_PARAM_NAMES, MODEL_STATE_NAMES, solve_model

_STATE_PREFIX = 'state_'
_RTOL = 1e-6
_ATOL = 1e-8


class ABMFitter:
    """Fit Lanchester-type ODE models to ABM simulation output.

    Supports multi-start optimization, automatic parameter scaling,
    and convergence diagnostics for robust fitting.
    """

    def __init__(self, model_func: Callable, param_names: list[str],
                 state_names: list[str],
                 state_mapping: Optional[dict[str, str]] = None):
        self.model_func = model_func
        self.param_names = list(param_names)
        self.state_names = list(state_names)
        self.state_mapping = state_mapping or {}
        self.n_params = len(self.param_names)
        self.n_states = len(self.state_names)
        self.fitted_params_: Optional[np.ndarray] = None
        self.fitted_param_dict: dict[str, float] = {}
        self.fit_result: Optional[Any] = None
        self.r_squared: Optional[float] = None
        self._t: Optional[np.ndarray] = None
        self._y_true: Optional[np.ndarray] = None
        self._y0: Optional[np.ndarray] = None

    # ── Data ────────────────────────────────────────────────────────────

    def _resolve_col(self, state_name: str) -> str:
        if state_name in self.state_mapping:
            return self.state_mapping[state_name]
        return f'{_STATE_PREFIX}{state_name}'

    def _extract_agent_series(self, df: pd.DataFrame,
                              agent_id: str) -> tuple[np.ndarray, np.ndarray]:
        sub = df[df['agent_id'] == agent_id].sort_values('time')
        t = sub['time'].values.astype(float)
        y = np.column_stack(
            [sub[self._resolve_col(s)].values.astype(float) for s in self.state_names]
        )
        return t, y

    def _extract_aggregate_series(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        col_map = {s: self._resolve_col(s) for s in self.state_names}
        for c in col_map.values():
            if c not in df.columns:
                raise KeyError(f'Column "{c}" not found. Available: {list(df.columns)}')
        grouped = df.groupby('time')
        t = np.array(sorted(grouped.groups.keys()), dtype=float)
        rows = []
        for ti in t:
            g = grouped.get_group(ti)
            rows.append([g[col_map[s]].mean() for s in self.state_names])
        return t, np.array(rows, dtype=float)

    def _validate_data(self, t: np.ndarray, y: np.ndarray) -> None:
        if len(t) < 10:
            raise ValueError(f'Need at least 10 time points, got {len(t)}.')
        if np.any(np.isnan(y)):
            raise ValueError('Data contains NaN values.')
        if np.ptp(y, axis=0).max() < 1e-5:
            warnings.warn('Data has near-zero variance — fitting may be unreliable.')

    # ── Objective ────────────────────────────────────────────────────────

    def _objective(self, params: np.ndarray, t: np.ndarray,
                   y_true: np.ndarray, y0: np.ndarray) -> float:
        def rhs(t_, y_):
            return np.array(self.model_func(t_, list(y_), *params))
        try:
            sol = solve_ivp(rhs, [t[0], t[-1]], y0, t_eval=t,
                            method='RK45', rtol=_RTOL, atol=_ATOL,
                            max_step=np.diff(t).mean())
            if not sol.success:
                return 1e12
            y_pred = sol.y.T
            if y_pred.shape != y_true.shape:
                return 1e12
            mse = np.mean((y_pred - y_true) ** 2)
            return float(1e12 if (np.isnan(mse) or np.isinf(mse)) else mse)
        except Exception:
            return 1e12

    # ── Single-start fit ─────────────────────────────────────────────────

    def fit_from_dataframe(
        self,
        df: pd.DataFrame,
        agent_id: Optional[str] = None,
        p0: Optional[list[float]] = None,
        bounds: Optional[list[tuple[float, float]]] = None,
        method: str = 'L-BFGS-B',
    ) -> Any:
        """Single-start fit.  See fit_robust() for production use."""
        if agent_id is not None:
            t, y_true = self._extract_agent_series(df, agent_id)
        else:
            t, y_true = self._extract_aggregate_series(df)
        self._validate_data(t, y_true)
        self._t, self._y_true, self._y0 = t, y_true, y_true[0]

        n = self.n_params
        if p0 is None:
            p0 = [0.5] * n
        if bounds is None:
            bounds = [(1e-4, 5.0)] * n

        result = minimize(self._objective, p0, args=(t, y_true, y_true[0]),
                          method=method, bounds=bounds,
                          options={'maxiter': 5000, 'ftol': 1e-12})
        self._save_result(result, y_true, t)
        return result

    # ── Multi-start fit (stable) ─────────────────────────────────────────

    def fit_robust(
        self,
        df: pd.DataFrame,
        agent_id: Optional[str] = None,
        bounds: Optional[list[tuple[float, float]]] = None,
        n_starts: int = 8,
        seed: int = 42,
    ) -> Any:
        """Multi-start fitting — tries several random initial guesses, returns the best.

        This is the recommended method for production use.
        """
        if agent_id is not None:
            t, y_true = self._extract_agent_series(df, agent_id)
        else:
            t, y_true = self._extract_aggregate_series(df)
        self._validate_data(t, y_true)
        self._t, self._y_true, self._y0 = t, y_true, y_true[0]

        n = self.n_params
        if bounds is None:
            bounds = [(1e-4, 5.0)] * n

        rng = np.random.RandomState(seed)
        best_result = None
        best_r2 = -np.inf

        for i in range(n_starts):
            p0 = [rng.uniform(b[0], b[1]) for b in bounds]
            result = minimize(self._objective, p0, args=(t, y_true, y_true[0]),
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 3000, 'ftol': 1e-12})
            y_mean = np.mean(y_true, axis=0)
            ss_total = np.sum((y_true - y_mean) ** 2)
            try:
                sol = solve_ivp(
                    lambda t_, y_: np.array(self.model_func(t_, list(y_), *result.x)),
                    [t[0], t[-1]], y_true[0], t_eval=t, method='RK45',
                    rtol=_RTOL, atol=_ATOL, max_step=np.diff(t).mean(),
                )
                if sol.success and sol.y.shape[1] == len(t):
                    y_pred = sol.y.T
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    r2 = 1.0 - ss_res / ss_total if ss_total > 0 else 0.0
                else:
                    r2 = -np.inf
            except Exception:
                r2 = -np.inf

            if r2 > best_r2:
                best_r2 = r2
                best_result = result

        if best_result is None:
            raise RuntimeError('All fitting attempts failed.')

        self._save_result(best_result, y_true, t)
        return best_result

    def fit_global(
        self,
        df: pd.DataFrame,
        agent_id: Optional[str] = None,
        bounds: Optional[list[tuple[float, float]]] = None,
    ) -> Any:
        """Global optimization via differential evolution (slow but thorough)."""
        if agent_id is not None:
            t, y_true = self._extract_agent_series(df, agent_id)
        else:
            t, y_true = self._extract_aggregate_series(df)
        self._validate_data(t, y_true)
        self._t, self._y_true, self._y0 = t, y_true, y_true[0]

        n = self.n_params
        if bounds is None:
            bounds = [(1e-4, 5.0)] * n

        result = differential_evolution(
            self._objective, bounds, args=(t, y_true, y_true[0]),
            seed=42, maxiter=1000, tol=1e-8, polish=True,
        )
        self._save_result(result, y_true, t)
        return result

    # ── Internals ────────────────────────────────────────────────────────

    def _save_result(self, result: Any, y_true: np.ndarray, t: np.ndarray) -> None:
        self.fitted_params_ = result.x
        self.fitted_param_dict = dict(zip(self.param_names, result.x))
        self.fit_result = result

        y_mean = np.mean(y_true, axis=0)
        ss_total = np.sum((y_true - y_mean) ** 2)
        if ss_total == 0:
            self.r_squared = 0.0
            return
        try:
            sol = solve_ivp(
                lambda t_, y_: np.array(self.model_func(t_, list(y_), *result.x)),
                [t[0], t[-1]], y_true[0], t_eval=t, method='RK45',
                rtol=_RTOL, atol=_ATOL, max_step=np.diff(t).mean(),
            )
            if sol.success and sol.y.shape[1] == len(t):
                y_pred = sol.y.T
                ss_res = np.sum((y_true - y_pred) ** 2)
                self.r_squared = max(0.0, 1.0 - ss_res / ss_total)
            else:
                self.r_squared = None
        except Exception:
            self.r_squared = None

    # ── Prediction ───────────────────────────────────────────────────────

    def predict(self, t: np.ndarray, y0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.fitted_params_ is None:
            raise RuntimeError('Call fit_*() before predict().')
        return solve_model(self.model_func, y0.tolist(), t,
                           self.fitted_params_.tolist())

    # ── Diagnostics ──────────────────────────────────────────────────────

    @property
    def converged(self) -> bool:
        return (self.fit_result is not None and
                self.fit_result.success and
                self.r_squared is not None and
                self.r_squared > 0.0)

    def summary_json(self) -> dict[str, Any]:
        r2 = None
        if self.r_squared is not None:
            try:
                r2 = round(float(self.r_squared), 6)
            except (TypeError, ValueError):
                r2 = float(self.r_squared)
        fun_val = None
        if self.fit_result is not None:
            try:
                fun_val = float(self.fit_result.fun)
            except (TypeError, ValueError):
                fun_val = None
        params = {}
        for k, v in self.fitted_param_dict.items():
            try:
                params[k] = round(float(v), 6)
            except (TypeError, ValueError):
                params[k] = str(v)
        return {
            'model': getattr(self.model_func, '__name__', 'model'),
            'params': params,
            'r_squared': r2,
            'converged': bool(self.converged),
            'n_params': self.n_params,
            'n_states': self.n_states,
            'state_names': self.state_names,
            'fun': fun_val,
        }

    def summary(self) -> str:
        j = self.summary_json()
        lines = [f'Model: {j["model"]}',
                 f'States: {j["state_names"]}',
                 f'Params: {j["params"]}',
                 f'R^2:    {j["r_squared"]}',
                 f'Converged: {j["converged"]}']
        return '\n'.join(lines)

    def __repr__(self) -> str:
        fitted = self.fitted_params_ is not None
        return f'ABMFitter({self.model_func.__name__}, fitted={fitted})'


# ── Builder ─────────────────────────────────────────────────────────────────

def make_fitter(model_name: str, state_mapping: Optional[dict[str, str]] = None,
                **param_fix: float) -> ABMFitter:
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f'Unknown model "{model_name}". Choose: {list(MODEL_REGISTRY)}')
    all_params = list(MODEL_PARAM_NAMES[model_name])
    state_names = list(MODEL_STATE_NAMES[model_name])
    model_func = MODEL_REGISTRY[model_name]
    for k in param_fix:
        if k in all_params:
            all_params.remove(k)
    if state_mapping is None:
        state_mapping = {}

    if param_fix:
        class _FixedModel:
            def __init__(self, func, fixed):
                self.func = func
                self.fixed = fixed
                self.__name__ = getattr(func, '__name__', 'model')
            def __call__(self, t, y, *free_params):
                merged = {}
                idx = 0
                for p in MODEL_PARAM_NAMES[model_name]:
                    merged[p] = self.fixed[p] if p in self.fixed else free_params[idx]
                    if p not in self.fixed:
                        idx += 1
                return self.func(t, y, **merged)
        wrapped = _FixedModel(model_func, param_fix)
    else:
        wrapped = model_func

    return ABMFitter(wrapped, all_params, state_names, state_mapping)


def compare_models(
    df: pd.DataFrame,
    model_names: list[str],
    agent_id: Optional[str] = None,
    robust: bool = True,
    **fit_kwargs,
) -> dict[str, ABMFitter]:
    results = {}
    for name in model_names:
        fitter = make_fitter(name)
        if robust:
            fitter.fit_robust(df, agent_id=agent_id, **fit_kwargs)
        else:
            fitter.fit_from_dataframe(df, agent_id=agent_id, **fit_kwargs)
        results[name] = fitter
    return results
