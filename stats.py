from pandas.core.frame import DataFrame
from scipy import stats
import pandas as pd

from .dataframe import DataFrame

class Stats(object):
    def __init__(self, dataframe_obj: DataFrame, whole_df: pd.DataFrame, sdf: pd.DataFrame, ddf: pd.DataFrame):
        self.dataframe = dataframe_obj
        self.__df = whole_df
        self.__sdf = sdf
        self.__ddf = ddf

    def set_df(self, df) -> None:
        self.__df = df
        self.__sdf = self.dataframe.get_sub_sdf(df)
        self.__ddf = self.dataframe.get_sub_ddf(df)

    # Get the number of medical records (ssd / dsd / total)
    def doc_counts(self) -> pd.DataFrame:
        categories = ["SSD", "DSD", "All"]
        doc_counts = [
            self.__sdf.groupby("DocLabel").ngroups,
            self.__ddf.groupby("DocLabel").ngroups,
            self.__df.groupby("DocLabel").ngroups
        ]
        return pd.DataFrame(index=categories, data={"doc_counts": doc_counts})

    # Perform Fisher's exact test on a keyword's frequency between two DataFrames (df2 is usually reference df - df1)
    @staticmethod
    def test_kw_rel(keyword, df1, df2, mode="greater", test="fisher"):
        counts = []

        for df in [df1, df2]:
            all_doc_count = df.groupby("DocLabel").ngroups
            kw_doc_count = df[df["Content"] == keyword].groupby("DocLabel").ngroups

            counts.append([kw_doc_count, all_doc_count - kw_doc_count])

        if test == "fisher":
            result = stats.fisher_exact(counts, alternative=mode)
        elif test == "chi":
            result = stats.chi2_contingency(counts)
        else:
            ValueError("arg test must be 'fisher' or 'chi'")
        
        return result # Return value: (oddsratio, p_value)