import matplotlib.pyplot as plt


def scatter_plot(X_train, X_test, y_train, y_test, title="Title"):
    plt.figure(figsize=(20, 7))
    plt.scatter(X_train, y_train, color='blue', label='train')
    plt.scatter(X_test, y_test, color='red', label='test')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('New deaths')
    plt.legend()
    plt.show()
