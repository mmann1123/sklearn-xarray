#%%
import numpy as np
import xarray as xr
from xarray.testing import assert_equal, assert_allclose
import numpy.testing as npt

from sklearn_xarray import wrap

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, KernelCenterer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

from tests.mocks import (
    DummyEstimator,
    DummyTransformer,
    ReshapingEstimator,
)
#%%
X = xr.Dataset(
            {
                "var_2d": (["sample", "feat_1"], np.random.random((100, 10))),
                "var_3d": (
                    ["sample", "feat_1", "feat_2"],
                    np.random.random((100, 10, 10)),
                ),
            },
            {
                "sample": range(100),
                "feat_1": range(10),
                "feat_2": range(10),
                "dummy": (["sample", "feat_1"], np.random.random((100, 10))),
            },
        )

estimator = wrap(
ReshapingEstimator(new_shape=(-1, 2)), reshapes="feat_1"
)

# test DataArray
X_da = X.var_3d

y = X_da[:, :2].drop_vars("feat_1")
y["dummy"] = y.dummy[:, 0]

estimator.fit(X_da)
yp = estimator.transform(X_da)

assert_allclose(yp, y)

# test Dataset
X_ds = X.var_2d.to_dataset()

y = X_ds.var_2d[:, :2].drop_vars("feat_1")
y["dummy"] = y.dummy[:, 0]

estimator.fit(X_ds)
yp = estimator.transform(X_ds).var_2d
# %%
