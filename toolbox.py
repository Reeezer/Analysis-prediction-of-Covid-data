import sklearn.metrics as metrics
import numpy as np
import plotly.express as ex
import plotly.graph_objects as go


def scatter_plot_squeeze(X_train, X_test, X_range, X_pred, y_train, y_test, y_range, y_pred, title="Title"):
    scatter_plot(X_train.squeeze(), X_test.squeeze(), X_range.squeeze(), X_pred.squeeze(), y_train, y_test, y_range, y_pred, title)


def scatter_plot(X_train, X_test, X_range, X_pred, y_train, y_test, y_range, y_pred, title="Title"):
    fig = go.Figure([
        go.Scatter(x=X_train, y=y_train, name='train', mode='markers', opacity=0.7),
        go.Scatter(x=X_test, y=y_test, name='test', mode='markers', opacity=0.7),
        go.Scatter(x=X_range, y=y_range, name='model fit'),
        go.Scatter(x=X_pred, y=y_pred, name='prediction')
    ])

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


def features_importance(model, X):
    colors = ['Positive' if c > 0 else 'Negative' for c in model.features_importance]
    fig = ex.bar(
        x=X.columns, y=model.features_importance, color=colors,
        color_discrete_sequence=['red', 'blue'],
        labels=dict(x='Feature', y='Linear coefficient'),
        title='Weight of each feature'
    )
    fig.show()


def regression_results(y_true, y_pred):
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mean_squared_error = metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance, 4))
    print('mean_squared_log_error: ', round(mean_squared_log_error, 4))
    print('mean_squared_absolute_error: ', round(median_absolute_error, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('MSE: ', round(mean_squared_error, 4))
    print('RMSE: ', round(np.sqrt(mean_squared_error), 4))
