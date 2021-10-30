import matplotlib.pyplot as plt


def scatter_plot(X_train, X_test, X_pred, y_train, y_test, y_pred, title="Title"):
    plt.figure(figsize=(20, 7))
    plt.scatter(X_train, y_train, color='blue', label='train')
    plt.scatter(X_test, y_test, color='red', label='test')
    plt.scatter(X_pred, y_pred, color='green', label='prediction')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('New deaths')
    plt.legend()
    plt.show()
