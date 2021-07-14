from scipy import stats

class Stats(object):
    def __init__(self):
        pass

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