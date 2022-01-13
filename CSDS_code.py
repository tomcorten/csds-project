from sklearn import datasets


def main():

    housing, target = datasets.fetch_openml(name="house_prices", as_frame=True, return_X_y=True)
    print(housing)
    print(target)


if __name__ == "__main__":
    main()