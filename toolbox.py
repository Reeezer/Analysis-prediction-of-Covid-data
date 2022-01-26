import sklearn.metrics as metrics
import numpy as np
import plotly.express as ex
import plotly.graph_objects as go


def scatter_plot(X_train, X_test, X_range, X_pred, y_train, y_test, y_range, y_pred, y_baseline=None, title="Title"):
    fig = go.Figure([
        go.Scatter(x=X_train, y=y_train, name='train', mode='markers', opacity=0.7),
        go.Scatter(x=X_test, y=y_test, name='test', mode='markers', opacity=0.7),
        go.Scatter(x=X_range, y=y_range, name='model fit'),
        go.Scatter(x=X_pred, y=y_pred, name='prediction')
    ])

    if y_baseline is not None:
        fig.add_scatter(x=y_baseline.index, y=y_baseline, opacity=0.3, name='baseline')

    fig.update_layout(
        title={
            'text': title,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Date',
        yaxis_title='New deaths'
    )

    fig.show()


def regression_results(y_true, y_pred, y_baseline=None):
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mean_squared_error = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    mean_absolute_error_baseline = metrics.mean_absolute_error(y_true, y_baseline)
    mean_squared_error_baseline = metrics.mean_squared_error(y_true, y_baseline)
    r2_baseline = metrics.r2_score(y_true, y_baseline)

    print(f'r2: {r2:.2f}, baseline: {r2_baseline:.2f}')
    print(f'Mean Absolute Error (MAE): {mean_absolute_error:.2f}, baseline: {mean_absolute_error_baseline:.2f}')
    print(f'Mean Square Error (MSE): {mean_squared_error:.2f}, baseline: {mean_squared_error_baseline:.2f}')
    print(f'Root Mean Square Error (RMSE): {np.sqrt(mean_squared_error):.2f}, baseline: {np.sqrt(mean_squared_error_baseline):.2f}')
