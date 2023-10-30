import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from perceptron_class import Perceptron
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split

def train_perceptron_and_predict(df, test_size=0.7, epochs=100_000):
    X = df[["sepal_length", "sepal_width"]].iloc[:100].values
    y = df.iloc[0:100].species.values
    y = np.where(y == 'setosa', -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    clf = Perceptron(epochs=epochs)
    clf.train(X_train, y_train)

    y_test_predicted = clf.predict(X_test)

    accuracy = np.mean(y_test_predicted == y_test)
    print(f"Acurácia na previsão dos dados de teste ({test_size * 100}%): {accuracy * 100:.2f}%")

    plt.figure(figsize=(10, 8))
    plot_decision_regions(X_train, y_train, clf=clf)
    plt.title(f"Decision Regions Plot ({test_size * 100}% Training Data)", fontsize=18)
    plt.xlabel("sepal length [cm]", fontsize=15)
    plt.ylabel("sepal width [cm]", fontsize=15)
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(clf.errors_)+1), clf.errors_, marker="o", label="error plot")
    plt.xlabel("Epochs")
    plt.ylabel("Missclassifications")
    plt.legend()
    plt.show()

    return clf, X_test, y_test, y_test_predicted

#------------------------------------------------------------------------#

df = pd.read_csv("D:\\Projetos\\bio_insp_codes\\trabalho2\\dataset\\iris_dataset.csv", sep=",")

proportions = [0.3, 0.5, 0.7]
for proportion in proportions:
    clf, X_test, y_test, y_test_predicted = train_perceptron_and_predict(df, test_size=proportion, epochs=100_000)

    print(f"Predições dos dados restantes ({(1 - proportion) * 100}%): {y_test_predicted}")
