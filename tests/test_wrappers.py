#%%
import unittest
import numpy as np
import xarray as xr
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn_xarray.common.wrappers import EstimatorWrapper
 
estimator = RandomForestClassifier(n_estimators=10)
wrapper = EstimatorWrapper(estimator=estimator) # this is the issue
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
hasattr(wrapper, 'estimator_')
#%%


# import numpy as np
# import xarray as xr
# from xarray.testing import assert_equal, assert_allclose
# import numpy.testing as npt

# from sklearn_xarray import wrap

# from sklearn.base import clone
# from sklearn.preprocessing import StandardScaler, KernelCenterer
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.svm import SVC

# from tests.mocks import (
#     DummyEstimator,
#     DummyTransformer,
#     ReshapingEstimator,
# )
 
# X = xr.Dataset(
#             {
#                 "var_2d": (["sample", "feat_1"], np.random.random((100, 10))),
#                 "var_3d": (
#                     ["sample", "feat_1", "feat_2"],
#                     np.random.random((100, 10, 10)),
#                 ),
#             },
#             {
#                 "sample": range(100),
#                 "feat_1": range(10),
#                 "feat_2": range(10),
#                 "dummy": (["sample", "feat_1"], np.random.random((100, 10))),
#             },
#         )

# estimator = wrap(
# ReshapingEstimator(new_shape=(-1, 2)), reshapes="feat_1"
# )

# # test DataArray
# X_da = X.var_3d

# y = X_da[:, :2].drop_vars("feat_1")
# y["dummy"] = y.dummy[:, 0]

# estimator.fit(X_da)
# yp = estimator.transform(X_da)

# assert_allclose(yp, y)

# # test Dataset
# X_ds = X.var_2d.to_dataset()

# y = X_ds.var_2d[:, :2].drop_vars("feat_1")
# y["dummy"] = y.dummy[:, 0]

# estimator.fit(X_ds)
# yp = estimator.transform(X_ds).var_2d
# %%

#%%
# import geowombat as gw
# import geopandas as gpd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestClassifier

# # from sklearn.model_selection import GridSearchCV, KFold
# from sklearn.naive_bayes import GaussianNB
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# # from sklearn_xarray.model_selection import CrossValidatorWrapper
# # from sklearn_xarray.preprocessing import Featurizer
# from xarray import DataArray as xr_da

# import geowombat as gw
# from geowombat.data import (
#     l8_224078_20200518,
#     l8_224078_20200518_points,
#     l8_224078_20200518_polygons,
# )
# from geowombat.ml import fit, fit_predict, predict

# aoi_point = gpd.read_file(l8_224078_20200518_points)
# aoi_point["lc"] = LabelEncoder().fit_transform(aoi_point.name)
# aoi_point = aoi_point.drop(columns=["name"])

# aoi_poly = gpd.read_file(l8_224078_20200518_polygons)
# aoi_poly["lc"] = LabelEncoder().fit_transform(aoi_poly.name)
# aoi_poly = aoi_poly.drop(columns=["name"])

# pl_wo_feat = Pipeline(
#     [
#         ("scaler", StandardScaler()),
#         ("pca", PCA()),
#         ("clf", GaussianNB()),
#     ]
# )
# pl_wo_feat_pca1 = Pipeline(
#     [
#         ("scaler", StandardScaler()),
#         ("pca", PCA(n_components=1)),
#         ("clf", GaussianNB()),
#     ]
# )

# tree_pipeline = Pipeline(
#     [
#         ("scaler", StandardScaler()),
#         ("clf", RandomForestClassifier(random_state=0)),
#     ]
# )

# cl_wo_feat = Pipeline(
#     [
#         ("scaler", StandardScaler()),
#         ("clf", KMeans(random_state=0)),
#     ]
# )


# with gw.config.update(
#             ref_res=300,
#         ):
#             with gw.open(l8_224078_20200518, nodata=0) as src:
#                     X, Xy, clf = fit(src, tree_pipeline, aoi_poly, col="lc")
#                     y1 = predict(src, X, clf)
#                     y2 = fit_predict(src, tree_pipeline, aoi_poly, col="lc")
                    
# print(np.all(np.isnan(y1.values[0, 0:5, 0])))
# print(np.all(np.isnan(y2.values[0, 0:5, 0])))
# print(
#     np.allclose(
#         y1.values,
#         y2.values,
#         equal_nan=True,
#     )
# )
#%%


#%%

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