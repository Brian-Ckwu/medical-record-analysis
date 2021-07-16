from matplotlib.ticker import PercentFormatter
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .dataframe import DataFrame
from .stats import Stats

class Plot(object):
    def __init__(self, labels: dict, dataframe_obj: DataFrame, stats_obj: Stats):
        self.lables = labels
        self.dataframe = dataframe_obj
        self.__df = self.dataframe.get_whole()
        self.__sdf = self.dataframe.get_ssd()
        self.__ddf = self.dataframe.get_dsd()
        self.stats = stats_obj

    def set_df(self, df) -> None:
        self.__df = df
        self.__sdf = self.dataframe.get_sub_sdf(df)
        self.__ddf = self.dataframe.get_sub_ddf(df)
    
    def doc_counts(self):
        doc_counts = self.stats.doc_counts().loc[["SSD", "DSD"]]
        return doc_counts.plot.bar()

    def pos_neg_prop(self):
        props = self.stats.pos_neg_prop()
        return props.plot.pie(subplots=True, autopct="%1.2f%%")