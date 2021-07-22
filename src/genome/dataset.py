import pandas as pd


class Dataset(object):
    def __init__(self, name, dataset_filename, dataset_sep):
        self.name = name
        self.pandasData = self.load_data(dataset_filename, dataset_sep)

        assert isinstance(self.pandasData, type(pd.DataFrame()))

    def exclude_columns(self, *args, inplace=False):
        if inplace:
            for x in args:
                self.pandasData.drop(columns=[x], inplace=inplace)
        else:
            df = self.pandasData
            for x in args:
                df = df.drop(columns=[x], inplace=inplace)
            return df

    def scale_column(self, column, scale, inplace=False):
        if inplace:
            self.pandasData[column] = self.pandasData[column] * scale
        else:
            return self.pandasData[column] * scale

    def load_data(self, filename, sep=","):
        return pd.read_csv(filename, sep=sep)
