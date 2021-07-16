import pandas as pd
import numpy as np

class DataFrame(object):
    def __init__(self, data_path):
        self.__df = pd.read_csv(data_path, index_col="index", encoding="utf-8", dtype={"Content": np.str_, "ICD9": np.str_})
        self.__df["posOrNeg"].fillna(3.0, inplace=True)
    
    """
        Get DataFrame functions
    """
    # 1. Get subdf from self.__df
    # Get the whole DataFrame
    def get_whole(self):
        return self.__df.copy() # avoid modification of self.__df from the user
    
    # Get the SSD DataFrame
    def get_ssd(self):
        return self.__df.loc[(self.__df["ICD9"] >= "780") & (self.__df["ICD9"] < "800")].copy() # DataFrame of symptom_dx

    # Get the DSD DataFrame
    def get_dsd(self):
        return self.__df.loc[(self.__df["ICD9"] < "780") | (self.__df["ICD9"] >= "800")].copy() # DataFrame of disease_dx

    # Get DataFrame of a particular TTAS code
    def get_cc(self, ttas_code):
        return self.__df.loc[self.__df["CC"].str.contains(ttas_code)].copy()

    # Get DataFrame of an icdcode
    def get_icd(self, icdcode):
        return self.__df.loc[self.__df["ICD9"] == icdcode].copy()

    # 2. Get subdf from a specific df
    @staticmethod
    def get_sub_sdf(df):
        return df.loc[(df["ICD9"] >= "780") & (df["ICD9"] < "800")]
    
    @staticmethod
    def get_sub_ddf(df):
        return df.loc[(df["ICD9"] < "780") | (df["ICD9"] >= "800")]

    """
        Change Dataframe functions: change, group, or split keywords in the passed DataFrame
    """
    @staticmethod
    def change_kw(df: pd.DataFrame, old_to_new: dict) -> pd.DataFrame:
        for old, new in old_to_new.items():
            df.loc[df["Content"] == old, "Content"] = new
            print(f"{old} -> {new}")
        return df
    
    @staticmethod
    def group_kw(df: pd.DataFrame, kw_groups: dict) -> pd.DataFrame:
        for ref_kw in kw_groups:
            kws = kw_groups[ref_kw] # keywords to be grouped
            for kw in kws:
                df.loc[df["Content"] == kw, "Content"] = ref_kw
                print(f"{kw} -> {ref_kw}")
            print("---------------------------------")
        return df
    
    @staticmethod
    def split_kw(df: pd.DataFrame, kws: list, delimiter: str) -> pd.DataFrame:
        for kw in kws:
            df.loc[df["Content"] == kw, "Content"] = df.loc[df["Content"] == kw, "Content"].str.split(delimiter)
        return df.explode("Content").reset_index(drop=True) # drop=True: avoid adding the old index as a column in the new DataFrame