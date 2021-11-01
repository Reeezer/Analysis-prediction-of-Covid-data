import matplotlib.pyplot as plt
import numpy as np

def scatter_plot(X_train, X_test, X_pred, y_train, y_test, y_pred, title="Title"):
    X1, y1 = reshape_order_data(X_train, y_train)
    X2, y2 = reshape_order_data(X_test, y_test)

    plt.figure(figsize=(20, 7))
    plt.fill_between(X1, y1, 0, color='blue', facecolor='cornflowerblue', alpha=0.6, label='train')
    plt.plot(X2, y2, color='red', linewidth=3, label='test')
    plt.plot(X_pred, y_pred, color='green', linewidth=3, label='prediction')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('New deaths')
    plt.legend()
    plt.show()

def reshape_order_data(X, y):
    X_reshaped = X.reshape(-1)
    y_reshaped = y.reshape(-1)
    order = np.argsort(X_reshaped)
    x_ordered = np.array(X_reshaped)[order]
    y_ordered = np.array(y_reshaped)[order]
    return x_ordered, y_ordered