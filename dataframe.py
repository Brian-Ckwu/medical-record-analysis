import pandas as pd
import numpy as np

class DataFrame(object):
    def __init__(self, data_path):
        self.__df = pd.read_csv(data_path, index_col="index", encoding="utf-8", dtype={"Content": np.str_, "ICD9": np.str_})
        self.__df["posOrNeg"].fillna(3.0, inplace=True)
    
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

    # 2. From a specific df
    @staticmethod
    def get_sub_sdf(df):
        return df.loc[(df["ICD9"] >= "780") & (df["ICD9"] < "800")]
    
    @staticmethod
    def get_sub_ddf(df):
        return df.loc[(df["ICD9"] < "780") | (df["ICD9"] >= "800")]