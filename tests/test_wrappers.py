#%%
import unittest
import numpy as np
import xarray as xr
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn_xarray.common.wrappers import EstimatorWrapper

class TestConfig(unittest.TestCase):

    def test_estimator_wrapper_initialization(self):
        estimator = DecisionTreeClassifier()
        wrapper = EstimatorWrapper(estimator=estimator)
        self.assertEqual(wrapper.estimator, estimator)

    def test_estimator_wrapper_fit_dataarray(self):
        X_da = xr.DataArray(
        np.random.random((100, 10)),
        coords={"sample": range(100), "feature": range(10)},
        dims=["sample", "feature"],
        )
        y_da = xr.DataArray(
            np.random.randint(0, 2, 100),
            coords={"sample": range(100)},
            dims=["sample"],
        )
        estimator = DecisionTreeClassifier()
        wrapper = EstimatorWrapper(estimator=estimator)
        wrapper.fit(X_da, y_da)
        self.assertTrue(hasattr(wrapper, 'estimator_'))

    def test_estimator_wrapper_fit_dataset(self):
        X_ds = xr.Dataset(
            {"var_1": (["sample", "feature"], np.random.random((100, 10)))},
            coords={"sample": range(100), "feature": range(10)},
        )
        y_ds = xr.DataArray(
            np.random.randint(0, 2, 100),
            coords={"sample": range(100)},
            dims=["sample"],
        )
        estimator = DecisionTreeClassifier()
        wrapper = EstimatorWrapper(estimator=estimator)
        wrapper.fit(X_ds, y_ds)
        self.assertTrue(hasattr(wrapper, 'estimator_dict_'))

    def test_estimator_wrapper_get_params(self):
        estimator = DecisionTreeClassifier(max_depth=3)
        wrapper = EstimatorWrapper(estimator=estimator)
        params = wrapper.get_params()
        self.assertEqual(params['max_depth'], 3)

    # def test_estimator_wrapper_set_params(self):
    #     estimator = DecisionTreeClassifier()
    #     wrapper = EstimatorWrapper(estimator=estimator)
    #     wrapper.set_params(max_depth=3)
    #     print(wrapper.estimator.max_depth)
    #     self.assertEqual(wrapper.estimator.max_depth, 3)

    def test_estimator_wrapper_getstate_setstate(self):
        estimator = DecisionTreeClassifier()
        wrapper = EstimatorWrapper(estimator=estimator)
        state = wrapper.__getstate__()
        new_wrapper = EstimatorWrapper(estimator=estimator)
        new_wrapper.__setstate__(state)
        self.assertEqual(new_wrapper.estimator, wrapper.estimator)

    def test_random_forest_classifier(self):
        estimator = RandomForestClassifier(n_estimators=10)
        wrapper = EstimatorWrapper(estimator=estimator)
        X_da = xr.DataArray(
            np.random.random((100, 10)),
            coords={"sample": range(100), "feature": range(10)},
            dims=["sample", "feature"],
        )
        y_da = xr.DataArray(
            np.random.randint(0, 2, 100),
            coords={"sample": range(100)},
            dims=["sample"],
        )
        wrapper.fit(X_da, y_da)
        self.assertTrue(hasattr(wrapper, 'estimator_'))

    def test_linear_regression(self):
        estimator = LinearRegression()
        wrapper = EstimatorWrapper(estimator=estimator)
        X_da = xr.DataArray(
            np.random.random((100, 10)),
            coords={"sample": range(100), "feature": range(10)},
            dims=["sample", "feature"],
        )
        y_da = xr.DataArray(
            np.random.random(100),
            coords={"sample": range(100)},
            dims=["sample"],
        )
        wrapper.fit(X_da, y_da)
        self.assertTrue(hasattr(wrapper, 'estimator_'))

if __name__ == '__main__':
    unittest.main()

def test_estimator_wrapper_get_params():
    estimator = DecisionTreeClassifier(max_depth=3)
    wrapper = EstimatorWrapper(estimator=estimator)
    params = wrapper.get_params()
    assert params['max_depth'] == 3

def test_estimator_wrapper_set_params():
    estimator = DecisionTreeClassifier()
    wrapper = EstimatorWrapper(estimator=estimator)
    wrapper.set_params(max_depth=3)
    assert wrapper.estimator.max_depth == 3

def test_estimator_wrapper_getstate_setstate():
    estimator = DecisionTreeClassifier()
    wrapper = EstimatorWrapper(estimator=estimator)
    state = wrapper.__getstate__()
    new_wrapper = EstimatorWrapper(estimator=estimator)
    new_wrapper.__setstate__(state)
    assert new_wrapper.estimator == wrapper.estimator