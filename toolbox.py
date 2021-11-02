import plotly.express as ex
import plotly.graph_objects as go

def scatter_plot(X_train, X_test, X_range, X_pred, y_train, y_test, y_range, y_pred, title="Title"):
    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers', opacity=0.7),
        go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers', opacity=0.7),
        go.Scatter(x=X_range.squeeze(), y=y_range, name='model fit'),
        go.Scatter(x=X_pred.squeeze(), y=y_pred, name='prediction')
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
    colors = ['Positive' if c > 0 else 'Negative' for c in model.coef_]
    fig = ex.bar(
        x=X.columns, y=model.coef_, color=colors,
        color_discrete_sequence=['red', 'blue'],
        labels=dict(x='Feature', y='Linear coefficient'),
        title='Weight of each feature'
    )
    fig.show()