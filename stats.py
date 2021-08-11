from scipy import stats
from pingouin import intraclass_corr
import pandas as pd
import math

from .dataframe import DataFrame

class Stats(object):
    def __init__(self, dataframe_obj: DataFrame, whole_df: pd.DataFrame, sdf: pd.DataFrame, ddf: pd.DataFrame, labels: dict):
        self.dataframe = dataframe_obj
        self.__df = whole_df
        self.__sdf = sdf
        self.__ddf = ddf
        self.__labels = labels

    def set_df(self, df: pd.DataFrame) -> None:
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
    
    # Get the proportion of the positive, negative, and unlabelled keywords
    def pos_neg_prop(self) -> pd.DataFrame:
        # Variables
        pos_neg_labels = [1.0, 2.0, 3.0]
        dx_types = ["SSD", "DSD"]
        dfs = [self.__sdf, self.__ddf]
        # Construct DataFrame
        prop_df = pd.DataFrame(index=pos_neg_labels, columns=dx_types)
        for dx_type, df in zip(dx_types, dfs):
            kw_count = df.groupby(["posOrNeg", "Content"])["DocLabel"].nunique().groupby(level=0).sum()
            prop_df[dx_type] = kw_count / kw_count.sum()

        return prop_df
    
    # Get the proportion of each keyword category (s/s, dx, drug, surg, non_surg, others)
    def kw_cat_prop(self) -> pd.DataFrame:
        categories = self.__labels
        dfs = {"SSD": self.__sdf, "DSD": self.__ddf}
        # Construct DataFrame
        prop_df = pd.DataFrame(index=categories.values(), columns=dfs.keys())
        for dx_type, df in dfs.items():
            doc_counts = df.groupby("DocLabel").ngroups
            kw_mean = dict()
            for cat_label, cat_name in categories.items():
                kw_df = df.loc[df["label"] == cat_label]
                nkw = kw_df.groupby("DocLabel")["Content"].nunique().sum()
                kw_mean[cat_name] = nkw / doc_counts
            prop_df[dx_type] = pd.Series(kw_mean)
            prop_df[dx_type] = prop_df[dx_type] / prop_df[dx_type].sum()

        return prop_df

    # Perform Fisher's exact test / Chi-square test on a keyword's frequency between two DataFrames (df2 is usually reference df - df1)
    @staticmethod
    def test_kw_rel(keyword: str, test_df: pd.DataFrame, comp_df: pd.DataFrame, test="fisher", mode="greater"):
        counts = []

        for df in [test_df, comp_df]:
            all_doc_count = df.groupby("DocLabel").ngroups
            kw_doc_count = df[df["Content"] == keyword].groupby("DocLabel").ngroups

            counts.append([kw_doc_count, all_doc_count - kw_doc_count])

        if test == "fisher":
            result = stats.fisher_exact(counts, alternative=mode)
        elif test == "chi2":
            result = stats.chi2_contingency(counts)
        else:
            ValueError("arg test must be 'fisher' or 'chi'")
        
        return result # Return value: (oddsratio, p_value)
    
    # Perform test_kw_rel() on a list of keywords
    @staticmethod
    def test_kws_rel(keywords, test_df: pd.DataFrame, comp_df: pd.DataFrame, test="fisher", mode="greater") -> pd.Series:
        kws_rel = pd.Series(index=keywords)
        # build series
        for keyword in keywords:
            test_result = Stats.test_kw_rel(keyword, test_df, comp_df, test, mode)
            p_value = test_result[1]
            kws_rel[keyword] = p_value

        return kws_rel
    
    # The proportion of keywords labelled cc_related / icd_related in a sub_df (e.g. ~90% of "epigastric pain"s are labelled cc_related in A0301)
    @staticmethod
    def related_kw_prop(keyword: str, sub_df: pd.DataFrame, target: str) -> float:
        kw_df = sub_df.loc[sub_df["Content"] == keyword]
        kw_total_count = kw_df["DocLabel"].nunique()
        cc_related_count = kw_df.loc[kw_df[f"{target.upper()}_Related"] == True]["DocLabel"].nunique()
        return cc_related_count / kw_total_count
    
    @staticmethod
    def related_kws_prop(keywords, sub_df: pd.DataFrame, target: str) -> pd.Series:
        kws_prop = pd.Series(index=keywords)
        for keyword in keywords:
            kws_prop[keyword] = Stats.related_kw_prop(keyword, sub_df, target)
        return kws_prop
    
    @staticmethod
    def c_tf_idf(keyword: str, class_df: pd.DataFrame, ref_df: pd.DataFrame) -> float:
        # calculate tf (scheme: log10(1 + keyword frequency)
        kw_count = class_df[class_df["Content"] == keyword]["DocLabel"].nunique()
        tf = math.log10(1 + kw_count) # +1: avoid the condition in which kw_count == 0
        # calculate idf
        doc_count = ref_df.groupby("DocLabel").ngroups
        kw_total_count = ref_df[ref_df["Content"] == keyword]["DocLabel"].nunique()
        idf = math.log10(doc_count / kw_total_count)
        # tfidf
        return (tf * idf, tf, idf)
    
    @staticmethod
    def c_tf_idf_kws(keywords, class_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.Series:
        s_tfidf = pd.Series(index=keywords)
        for keyword in keywords:
            s_tfidf[keyword] = Stats.c_tf_idf(keyword, class_df, ref_df)[0]
        return s_tfidf
    
    @staticmethod
    def icc(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        df = df.melt(id_vars=[id_col])
        return intraclass_corr(data=df, raters=id_col, targets="variable", ratings="value")
    
    @staticmethod
    def spearman_cor(a: list, b: list) -> tuple:
        return stats.spearmanr(a, b)