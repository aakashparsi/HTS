from collections import namedtuple

import numpy
import pandas
import pytest
# from fbprophet import Prophet
from pmdarima import AutoARIMA


from hts.model import AutoArimaModel, FBProphetModel, HoltWintersModel, SarimaxModel
from hts.model.base import TimeSeriesModel


# def test_instantiate_fb_model_uv(uv_tree):
#     fb = FBProphetModel(node=uv_tree)
#     assert isinstance(fb, TimeSeriesModel)
#     fb = FBProphetModel(node=uv_tree, capacity_max=1)
#     assert isinstance(fb, TimeSeriesModel)
#     fb = FBProphetModel(node=uv_tree, capacity_min=1)
#     assert isinstance(fb, TimeSeriesModel)


# def test_fit_predict_fb_model_mv(mv_tree):
#     exog = pandas.DataFrame({"precipitation": [1], "temp": [20]})
#     fb = FBProphetModel(node=mv_tree)
#     assert isinstance(fb, TimeSeriesModel)
#     fb.fit()
#     fb.predict(mv_tree, exogenous_df=exog)
#     assert isinstance(fb.forecast, pandas.DataFrame)
#     assert isinstance(fb.residual, numpy.ndarray)
#     assert isinstance(fb.mse, float)


# def test_fit_predict_fb_model_mv(mv_tree):
#     exog = pandas.DataFrame({"precipitation": [1, 2], "temp": [20, 30]})
#     fb = FBProphetModel(node=mv_tree)
#     assert isinstance(fb, TimeSeriesModel)
#     fb.fit()
#     fb.predict(mv_tree, exogenous_df=exog)
#     assert isinstance(fb.forecast, pandas.DataFrame)
#     assert isinstance(fb.residual, numpy.ndarray)
#     assert isinstance(fb.mse, float)


# def test_fit_predict_fb_model_uv(uv_tree):
#     fb = FBProphetModel(node=uv_tree)
#     fb.fit()
#     assert isinstance(fb.model, Prophet)
#     fb.predict(uv_tree)
#     assert isinstance(fb.forecast, pandas.DataFrame)
#     assert isinstance(fb.residual, numpy.ndarray)
#     assert isinstance(fb.mse, float)


def test_fit_predict_ar_model_mv(mv_tree):
    ar = AutoArimaModel(node=mv_tree)
    ar.fit(max_iter=1)
    assert isinstance(ar.model, AutoARIMA)
    exog = pandas.DataFrame({"precipitation": [1], "temp": [20]})
    ar.predict(mv_tree, steps_ahead=1, exogenous_df=exog)
    assert isinstance(ar.forecast, pandas.DataFrame)
    assert isinstance(ar.residual, numpy.ndarray)
    assert isinstance(ar.mse, float)


def test_fit_predict_ar_model_uv(uv_tree):
    ar = AutoArimaModel(
        node=uv_tree,
    )
    ar.fit(max_iter=1)
    assert isinstance(ar.model, AutoARIMA)
    ar.predict(uv_tree)
    assert isinstance(ar.forecast, pandas.DataFrame)
    assert isinstance(ar.residual, numpy.ndarray)
    assert isinstance(ar.mse, float)


def test_fit_predict_sarimax_model_uv(uv_tree):
    sar = SarimaxModel(
        node=uv_tree,
        max_iter=1,
    )
    fitted_sar = sar.fit()
    assert isinstance(fitted_sar, SarimaxModel)
    sar.predict(uv_tree)
    assert isinstance(sar.forecast, pandas.DataFrame)
    assert isinstance(sar.residual, numpy.ndarray)
    assert isinstance(sar.mse, float)


def test_fit_predict_hw_model_uv(uv_tree):
    hw = HoltWintersModel(
        node=uv_tree,
    )
    fitted_hw = hw.fit()
    assert isinstance(fitted_hw, HoltWintersModel)
    hw.predict(uv_tree)
    assert isinstance(hw.forecast, pandas.DataFrame)
    assert isinstance(hw.residual, numpy.ndarray)
    assert isinstance(hw.mse, float)


def test_fit_predict_hw_model_uv_with_transform(uv_tree):
    Transform = namedtuple("Transform", ["func", "inv_func"])
    transform_pos_neg = Transform(func=numpy.exp, inv_func=lambda x: -x)

    hw = HoltWintersModel(node=uv_tree, transform=transform_pos_neg)
    fitted_hw = hw.fit()
    assert isinstance(fitted_hw, HoltWintersModel)
    preds = hw.predict(uv_tree)
    assert not (preds.forecast.values > 0).any()

    assert isinstance(hw.forecast, pandas.DataFrame)
    assert isinstance(hw.residual, numpy.ndarray)
    assert isinstance(hw.mse, float)


def test_fit_predict_model_invalid_transform(uv_tree):
    Transform = namedtuple("Transform", ["func_invalid_arg", "inv_func"])
    transform_pos_neg = Transform(func_invalid_arg=numpy.exp, inv_func=lambda x: -x)
    with pytest.raises(ValueError):
        HoltWintersModel(node=uv_tree, transform=transform_pos_neg)


from hts.utilities.load_data import load_mobility_data
from hts.hierarchy import HierarchyTree
from hts.core import regressor
import pandas as pd
def test_ar_model():
    dummydf = pd.DataFrame(
    [
        [10, 1, 4, 5, 100, 10, 10, 110], 
        [20, 2, 8, 10, 25, 5, 5, 45],
        [30, 3, 12, 15, 400, 20, 20, 430],
        [40, 4, 16, 20, 225, 15, 15, 265],
    ],
    columns = ['target1', 'exog11', 'exog12', 'exog13', 'target2', 'exog21', 'exog22', 'total'],
    index = ['2021-01', '2021-02', '2021-03', '2021-04']        
    )
    dummydf.index = pd.to_datetime(dummydf.index)
    dummydf.index = dummydf.index.to_period('M')

    exogdf = pd.DataFrame(
    [
        [1, 16, 25, 2, 2],
        [4, 64, 100, 1, 1]
    ],
    columns = ['exog11', 'exog12', 'exog13', 'exog21', 'exog22'],
    index = ['2021-05', '2021-06']
    )

    exogdf.index = pd.to_datetime(exogdf.index)
    exogdf.index = exogdf.index.to_period('M')

    hier = {
            'total': ['target1', 'target2']
            }
    exogenous = {
            'target1': ['exog11', 'exog12', 'exog13'],
            'target2': ['exog21', 'exog22']
        }
    
    print(dummydf)
    # ht = HierarchyTree.from_nodes(hier, dummydf, exogenous=exogenous)
    htsmodel = regressor.HTSRegressor(model = 'auto_arima', revision_method = 'BU', n_jobs = 0)
    htsfit = htsmodel.fit(dummydf, hier, exogenous = exogenous) 
    print("fitting completed\n")
    pred = htsfit.predict(steps_ahead = 2, exogenous_df=exogdf)
    print(pred)
    return "DONE"