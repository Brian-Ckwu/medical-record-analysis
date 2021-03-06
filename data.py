# import standard or third-party libraries
from matplotlib.ticker import PercentFormatter
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

# import self-designed libraries (component classes)
from .dataframe import DataFrame
from .stats import Stats
from .plot import Plot

class Data(object):
    def __init__(self, data_path: str):
        # DataFrame object
        self.dataframe = DataFrame(data_path)
        # Classify the icdcodes as symptom_dx or disease_dx
        self.__df = self.dataframe.get_whole()
        self.__sdf = self.dataframe.get_ssd()
        self.__ddf = self.dataframe.get_dsd()

        # Keyword classification
        self.__labels = { # ordered dict
            6: "s/s",
            4: "dx",
            5: "drug",
            7: "surg",
            11: "non_surg",
            8: "others"
        }

        # Stats object
        self.stats = Stats(dataframe_obj=self.dataframe, whole_df=self.__df, sdf=self.__sdf, ddf=self.__ddf, labels=self.__labels)

        # Plot object
        self.plot = Plot(labels=self.__labels, dataframe_obj=self.dataframe, stats_obj=self.stats)

        # Store cc_codes list
        self.__cc_codes = []
        ccss = self.__df['CC'].unique()
        for ccs in ccss:
            for cc in ccs.split(';'):
                if (cc not in self.__cc_codes):
                    self.__cc_codes.append(cc)

        # Load auxiliary files for converting codes to names
        dirname = os.path.dirname(__file__)
        # ICD-9 code to diagnosis name
        icd_to_dx_path = os.path.join(dirname, "icd_to_dx\\allcodes.json")
        with open(icd_to_dx_path, mode="rt", encoding="utf-8") as f:
            self.icd_to_dx = json.load(f)
        # TTAS code to Chinese name
        ttas_path = os.path.join(dirname, "ttas\\ttas_code_to_name.json")
        with open(ttas_path, mode="rt", encoding="utf-8") as f:
            self.ttas_dict = json.load(f)
    
    def get_labels(self):
        return self.__labels.copy()
    
    def get_cc_codes(self):
        return self.__cc_codes.copy()
    
    def map_kw_to_cat(self) -> dict:
        # Get (kw, label) tuples sorted by row count
        whole_df = self.dataframe.get_whole()
        kw_label_tuples = whole_df.groupby(["Content", "label"]).size().sort_values(ascending=False).index
        # Map keyword to category
        kw2cat = dict()
        for kw, label in kw_label_tuples:
            if not kw2cat.get(kw):
                kw2cat[kw] = label
        return kw2cat

    """
        Get Stats Functions: getting the stats of the DataFrame
    """
    # Get doc counts stratified by age
    def get_age_doc_counts(self):
        def get_age_doc(df):
            return df.groupby('Age')['DocLabel'].nunique().groupby(lambda age: age // 10).sum()

        d = {}
        for name, df in zip(['SSD', 'DSD', 'All'], [self.__sdf, self.__ddf, self.__df]):
            doc_count = get_age_doc(df)
            doc_count.at[8] = doc_count[doc_count.index >= 8].sum()
            doc_count.drop([9, 10], inplace=True)
            d[name] = doc_count.values

        return pd.DataFrame(data=d, index=[f'{i}0-{i}9' for i in range(2, 8)] + ['80+'])
    
    def get_cc_doc_counts_new(self, mode="first") -> pd.DataFrame:
        return

    # Get doc counts of each cheif complaint
    def get_cc_doc_counts(self, num):
        # Function: get all cc codes
        def get_cc_codes(df):
            cc_codes = []
            ccss = df['CC'].unique()
            for ccs in ccss:
                for cc in ccs.split(';'):
                    if (cc not in cc_codes):
                        cc_codes.append(cc)
            return cc_codes
        # Function: get cc_doc_counts_dict from df
        def get_cc_doc_counts_dict(df):
            cc_doc_counts_dict = {}
            # Initiate cc_doc_counts_dict (assign each code with initial value of zero)
            cc_codes = get_cc_codes(df)
            for cc_code in cc_codes:
                cc_doc_counts_dict[cc_code] = 0
            # Count doc number of each cc_code
            ccs_doc_counts = df.groupby('CC')['DocLabel'].nunique()
            for codes, count in ccs_doc_counts.items():
                for code in codes.split(';'):
                    cc_doc_counts_dict[code] += count
            
            return cc_doc_counts_dict
        # Get cc_doc_counts_dict
        cc_doc_counts = {}
        for dx_type, df in zip(['SSD', 'DSD'], [self.__sdf, self.__ddf]):
            cc_doc_counts[dx_type] = get_cc_doc_counts_dict(df)
        # Construct DataFrame
        df = pd.DataFrame(data=cc_doc_counts).fillna(0)
        df['All'] = df['SSD'] + df['DSD']
        df = df.sort_values(by='All', ascending=False).head(num)
        ttas_names = [self.ttas_dict[code] for code in df.index]
        df['Names'] = ttas_names

        return df[['Names', 'SSD', 'DSD', 'All']].convert_dtypes()
    
    # Send the DataFrame you want to describe (e.g. s_df or d_df)
    def describe(self, df):
        # Initiate variables
        desc = {'ndoc': 0, 'kw_stat': {}}
        nkw_data = {'total': [], 'pos': [], 'neg': []}
        for label_name in self.__labels.values():
            nkw_data[label_name] = []        
        # Handling nkw_data
        doc_group = df.groupby('DocLabel')
        nkw_data['total'] = list(doc_group['Content'].nunique().to_numpy())
        for sdf in doc_group.__iter__():
            # pos or neg
            png = sdf[1].groupby('posOrNeg')
            for i, name in [(1.0, 'pos'), (2.0, 'neg')]:
                try:
                    nkw_data[name].append(png.get_group(i)['Content'].nunique())
                except KeyError:
                    nkw_data[name].append(0)
            # by kw categories
            catg = sdf[1].groupby('label')
            for label, label_name in self.__labels.items():
                try:
                    nkw_data[label_name].append(catg.get_group(label)['Content'].nunique())
                except KeyError:
                    nkw_data[label_name].append(0)
        # Handling desc
        desc['ndoc'] = df.groupby('DocLabel').ngroups
        for name, data in nkw_data.items():
            desc['kw_stat'][name] = {}
            desc['kw_stat'][name]['mean'] = np.mean(data)
            desc['kw_stat'][name]['std'] = np.std(data)
        return nkw_data, desc
    
    def get_mean_nkw(self):
        d = {}

        for dx_name, df in zip(['SSD', 'DSD', 'All'], [self.__sdf, self.__ddf, self.__df]):
            dd = {}

            # Process data
            # Number of medical records
            doc_count = df.groupby('DocLabel').ngroups
            # Mean of total keywords
            dd['total'] = df.groupby('DocLabel')['Content'].nunique().sum()
            # Mean of pos/neg/unlabelled keywords
            labels = {1.0: 'pos', 2.0: 'neg', 3.0: 'unlabelled'}
            for value, label in labels.items():
                dd[label] = df[df['posOrNeg'] == value].groupby('DocLabel')['Content'].nunique().sum()
            # Mean of different categories of keywords
            for value, label in self.__labels.items():
                dd[label] = df[df['label'] == value].groupby('DocLabel')['Content'].nunique().sum()
            
            # Construct series
            s = pd.Series(data=dd) / doc_count
            d[dx_name] = s
        
        return pd.DataFrame(data=d)

    def get_keywords_diff(self, pos_or_neg, category):
        # Count keywords number in SSD and DSD medical records
        kwn = {}
        for dx_type, df in zip(['SSD', 'DSD'], [self.__sdf, self.__ddf]):
            if ((pos_or_neg in [1, 2, 'total']) and (category in [4, 5, 6, 7, 8, 11])):
                if (pos_or_neg == 'total'):
                    df = df[df['label'] == category]
                else:
                    df = df[(df['posOrNeg'] == pos_or_neg) & (df['label'] == category)]
            else:
                raise ValueError('pos_or_neg must be 1, 2, or 3 / category msut be 4, 5, 6, 7, 8, or 11')
            kwn[dx_type] = df.groupby('Content')['DocLabel'].nunique() / df.groupby('DocLabel').ngroups
        # Construct DataFrame
        kwn_df = pd.DataFrame(data=kwn).fillna(0)
        kwn_df['Dif'] = kwn_df['SSD'] - kwn_df['DSD']
        return kwn_df.sort_values(by='Dif', ascending=False)

    def group_icdcodes_ndoc(self):
        return self.__df.groupby('ICD9')['DocLabel'].nunique().groupby(lambda code: str(code).split('.')[0].zfill(3)).sum()

    """
        Statistical tests functions
    """
    # Make keyword-cc relation (based on Fisher's exact test) DataFrame
    def make_kw_cc_rel_by_test(self, kw_num, cc_num, test="fisher"):
        # Get keywords
        kw_series = self.__df.groupby("Content")["DocLabel"].nunique()
        kws = kw_series.sort_values(ascending=False)[:kw_num].index
        # Get ccs
        ccs = self.get_cc_doc_counts(cc_num).index

        # Construct df
        df = pd.DataFrame(index=kws, columns=ccs, dtype=np.float64).fillna(0)

        # Loop through ccs
        for cc in ccs:
            cc_df = self.get_whole_df_from_cc(cc)
            not_cc_df = self.__df.drop(cc_df.index)
            for kw in kws:
                p_value = self.stats.test_kw_rel(kw, cc_df, not_cc_df, test=test)[1]
                df.at[kw, cc] = p_value
                print(f"{kw} in {cc}: {p_value:.2f}")
        
        return df
    
    # Make keyword-cc relation DataFrame (sort according to the ratio of cc_related count / all count of a keyword in a chief complaint)
    def make_kw_cc_rel_labelled(self, cc_code):
        # DataFrame of the chief complaint
        df = self.get_whole_df_from_cc(cc_code)

        # Get total count & cc_related count of the keywords
        kw_total_count = df.groupby("Content")["DocLabel"].nunique()
        kw_ccr_count = df[df["CC_Related"] == True].groupby("Content")["DocLabel"].nunique()

        # Concatenate two series and calculate the ratio
        kw_concat = pd.concat([kw_ccr_count, kw_total_count], axis=1).fillna(0).convert_dtypes()
        kw_concat.columns = ["cc_related", "all"]
        kw_concat["ratio"] = kw_concat["cc_related"] / kw_concat["all"]
        kw_concat.sort_values(by="ratio", ascending=False, inplace=True)

        return kw_concat

    """
        Names related functions
    """
    # new func to replace get_dx_names
    def get_dx_name(self, depth, icdcode):
        return self.icd_to_dx[str(depth)][str(icdcode)]

    # old func: convert icdcodes to diagnosis names
    def get_dx_names(self, icdcodes):
        names = []
        for icdcode in icdcodes:
            for codelst in self.icd_to_dx:
                try:
                    if codelst[2]["code"] == icdcode:
                        names.append(codelst[2]["descr"])
                        break
                except IndexError:
                    pass
        return names

    """
        Plotting functions
    """
    # Plot the diagnosis proportion of symptom_dx & disease_dx
    def plot_dx_prop(self, dx_num):
        grouped_ndoc = self.group_icdcodes_ndoc()
        s_data = grouped_ndoc[grouped_ndoc.index >= '780']
        s_data = (s_data / s_data.sum()).sort_values()[-dx_num:]
        d_data = grouped_ndoc[grouped_ndoc.index < '780']
        d_data = (d_data / d_data.sum()).sort_values()[-dx_num:]
        data = [
            {
                'dx_names': [f"{name} ({icd})" for name, icd in zip(self.get_dx_names(s_data.index), s_data.index)],
                'props': s_data.values,
            },
            {
                'dx_names': [f"{name} ({icd})" for name, icd in zip(self.get_dx_names(d_data.index), d_data.index)],
                'props': d_data.values,
            }
        ]

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs = axs.ravel()
        for i, d in enumerate(data):
            axs[i].set(xlim=(0, d['props'][-1] * 1.1), title='symptom_dx' if i == 0 else 'disease_dx')
            axs[i].barh(d['dx_names'], d['props'], color=f'C{i}')
            axs[i].xaxis.set_major_formatter(PercentFormatter(1))
            # Show the percentage to the right of each bar
            for j, prop in enumerate(d['props']):
                axs[i].text(prop + d['props'][-1] * 0.01, j - 0.1, f"{round(prop * 100, 2)}%")
    
    # Get diagnosis count based on the first 3 digits of icdcodes
    def get_dx_counts(self, num):
        dx_props = {}
        dx_types = ['SSD', 'DSD']
        for dx_type, df in zip(dx_types, [self.__sdf, self.__ddf]):
            dx_props[dx_type] = df.groupby('ICD9')['DocLabel'].nunique().groupby(lambda icd: icd[:3]).sum().sort_values(ascending=False).head(num)

        return [pd.DataFrame(data={'names': self.get_dx_names(dx_props[dx_type].index),'counts': dx_props[dx_type]}) for dx_type in dx_types]

    # Plot the keyword number distribution of every category
    def plot_nkw_dist_every_cat(self, s_data, d_data, bins): # Get s_desc_ & d_desc from self.describe function
        # Function for formatting p_value
        def format_p(p_value):
            if (p_value < 0.001):
                return 'p < 0.001'
            elif (p_value < 0.05):
                return 'p < 0.05'
            else:
                return f'p = {round(p_value, 3)}'
        # Plotting process
        fig, axs = plt.subplots(3, 3, figsize=(18, 12))
        axs = axs.ravel()
        categories = list(s_data.keys())
        for i in range(9):
            category = categories[i]
            # Plot distribution
            for j, data in enumerate([s_data, d_data]):  
                axs[i].hist(data[category], weights=np.ones(len(data[category])) / len(data[category]), bins=bins, label='symptom_dx' if j == 0 else 'disease_dx', alpha=0.3)
                axs[i].axvline(x=np.mean(data[category]), color=f'C{j}')
            # Show mean & p-value
            for j, data in enumerate([s_data, d_data]):
                axs[i].text(x=0.65 * axs[i].get_xlim()[1], y=(0.65 - j * 0.10) * axs[i].get_ylim()[1], s=f'{round(np.mean(data[category]), 2):.2f}??{round(np.std(data[category]), 2):.2f}', color=f'C{j}', fontsize=13.5)
            p_value = stats.ttest_ind(a=s_data[category], b=d_data[category], equal_var=False)[1]
            p_str = format_p(p_value)
            axs[i].text(x=0.65 * axs[i].get_xlim()[1], y=(0.65 - 2 * 0.10) * axs[i].get_ylim()[1], s=p_str, color='C2', fontsize=13.5)
            # Other formats
            axs[i].legend()
            axs[i].yaxis.set_major_formatter(PercentFormatter(1))
            axs[i].set_title(category)
    
    # Plot common keywords    
    def plot_common_kw(self, p_min, p_max=1, cat='total'): # cat: {4: 'diagnosis', 5: 'drug', 6: 's/s', 7: 'surgery', 8: 'others', 11: 'non_surgery'}
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
        fig.suptitle(('total_kw' if cat == 'total' else f'{self.__labels[cat]}_kw') + f' ({round(p_min * 100, 1)}% - {round(p_max * 100, 1)}%)', x=0.08, y=0.91, fontweight=600)
        for i, df in enumerate([self.__sdf, self.__ddf]):
            # Fill the empty values in 'posOrNeg' column with 3.0
            df = df.copy()
            if cat != 'total':
                df = df[df['label'] == cat]
            # Count the ndoc and find the keywords of which proportion is greater than p_cutoff
            ndoc = df['DocLabel'].nunique()
            kwprop = df.groupby(['Content', 'posOrNeg'])['DocLabel'].nunique() / ndoc
            kwprop_sum = kwprop.groupby('Content').sum().sort_values()
            kwprop_sum_inrange = kwprop_sum[(p_min <= kwprop_sum) & (kwprop_sum <= p_max)]
            indexes = kwprop_sum_inrange.index
            # Plot
            kwprop_df = kwprop.loc[indexes].unstack().fillna(0)
            kwprop_df.plot(kind='barh', width=0.8, stacked=True, ax=axs[i], title='symptom_dx' if i == 0 else 'disease_dx', xlim=(0, kwprop_sum_inrange[-1] * 1.1)).legend(title='posOrNeg', loc='lower right')
            axs[i].xaxis.set_major_formatter(PercentFormatter(1))
            for j, props in enumerate(kwprop_df.values):
                # Plot separate percentages
                left = 0
                for prop in props:
                    pos = left + prop / 2
                    left += prop
                    axs[i].text(pos, j, f'{round(prop * 100, 1)}' if prop > 0.015 else '', ha='center', va='center', color='white')
                # Plot total percentage
                axs[i].text(props.sum() + kwprop_df.values[-1].sum() * 0.01, j, f'{round(props.sum() * 100, 2)}%', va='center')

    def plot_kw_prop_in_icdcodes(self, kw, icdcodes):
        # Initiate variables
        df = self.__df
        df = df.loc[df['Content'] == kw]
        df.loc[:, 'posOrNeg'].fillna(value=3.0, inplace=True)
        icdcodes_p3_ndoc = self.group_icdcodes_ndoc()
        # Keyword count
        kw_count = df.groupby([lambda index: str(df.loc[index, 'ICD9']).split('.')[0].zfill(3), 'posOrNeg'])['DocLabel'].nunique().unstack().fillna(0)
        # Get ndoc of icdcodes
        icdcodes_ndoc = icdcodes_p3_ndoc[kw_count.index]
        # Keyword proportion
        kw_prop = kw_count.apply(lambda x: (x / icdcodes_ndoc[x.index]), axis=0)        
        # Handle empty icdcodes
        for icdcode in icdcodes:
            if (icdcode not in kw_prop.index):
                kw_prop.at[icdcode] = 0
        sorted_indexes = kw_prop.sum(axis=1)[icdcodes].sort_values().index
        # Plot
        for_plot = kw_prop.loc[sorted_indexes]
        for_plot.plot(kind='barh', width=0.8, stacked=True, title=kw, xlim=(0, for_plot.sum(axis=1)[-1] * 1.15)).legend(title='posOrNeg', loc='lower right')
        plt.yticks(ticks=range(len(sorted_indexes)), labels=[f"{name} ({icd}) (ndoc={ndoc})" for name, icd, ndoc in zip(self.get_dx_names(sorted_indexes), sorted_indexes, icdcodes_p3_ndoc[sorted_indexes])])
        for j, props in enumerate(for_plot.values):
            # Plot separate percentages
            left = 0
            for prop in props:
                pos = left + prop / 2
                left += prop
                plt.text(pos, j, f'{round(prop * 100, 1)}' if prop > 0.015 else '', ha='center', va='center', color='white')
            # Plot total percentage
            plt.text(props.sum() + for_plot.values[-1].sum() * 0.01, j, f'{round(props.sum() * 100, 2)}%', va='center')
    
    def plot_age_nkw_dist(self, target, category=0, interval=10, ax=None):
        if (interval not in [5, 10]):
            raise Exception('The argument interval can only be 5 or 10.')
        
        if (target not in ['sex', 'diagnosis']):
            raise Exception('The argument target can only be "sex" or "diagnosis".')

        # Extract DataFrame of certain category
        labels = self.__labels.copy()
        labels[0], labels[1], labels[2] = 'total', 'pos', 'neg'
        df = self.__df
        if (not category):
            cat_df = df
        else:
            if (category < 3):
                cat_df = df[df['posOrNeg'] == category]
            else:
                cat_df = df[df['label'] == category]

        # Data processing
        age_index = [f'{x * interval}-{(x + 1) * interval - 1}' for x in range(20 // interval, 110 // interval)]
        age_mkw_series = {}

        if (target == 'sex'):
            l_df, r_df = df[df['Sex'] == 'M'], df[df['Sex'] == 'F']
            l_cat_df, r_cat_df = cat_df[cat_df['Sex'] == 'M'], cat_df[cat_df['Sex'] == 'F']
            l_name, r_name = 'Male', 'Female'
        else:
            l_df, r_df = self.__sdf, self.__ddf
            l_cat_df, r_cat_df = self.dataframe.get_sub_sdf(cat_df), self.dataframe.get_sub_ddf(cat_df)
            l_name, r_name = 'Symptom_dx', 'Disease_dx'

        for i, df, cat_df in zip(range(2), [l_df, r_df], [l_cat_df, r_cat_df]):
            age_ndoc = df.set_index('Age').groupby(lambda age: age // interval)['DocLabel'].nunique()
            age_nkw = cat_df.groupby(['Age', 'DocLabel'])['Content'].nunique().groupby(lambda index: index[0] // interval).sum() * (-1) ** (i + 1)
            age_mkw = age_nkw.multiply(1 / age_ndoc, fill_value=0)
            age_mkw_series[i] = pd.Series(index=map(lambda x: f'{x * interval}-{(x + 1) * interval -1}', age_mkw.index), data=age_mkw.values)
        
        age_mkw_df = pd.DataFrame(data = {
            'Age': age_index,
            l_name: age_mkw_series[0],
            r_name: age_mkw_series[1]
        }, index=age_index).fillna(0)

        for i, target in enumerate([l_name, r_name]):
            sns.barplot(x=target, y='Age', data=age_mkw_df, order=reversed(age_index), color=f'C{i}', label=target, ax=ax)
            ax.set_xlabel('Number of keywords')
            ax.set_ylabel('Age')
            ax.set_title(f'Age-keywords distribution ({labels[category]})')
            ax.legend(loc='lower left')
    
    @staticmethod
    def get_sex_prop(df):
        sex_count = df.groupby('Sex')['DocLabel'].nunique()
        total_doc = df.groupby('DocLabel').ngroups
        return sex_count / total_doc
    
    @staticmethod
    def get_age_prop(df):
        age_count = df.groupby('Age')['DocLabel'].nunique().groupby(lambda age: age // 10).sum()
        total_doc = df.groupby('DocLabel').ngroups
        return age_count / total_doc
    
    @staticmethod
    def plot_age_dist_hist(df, bins):
        age_count = df.groupby('Age')['DocLabel'].nunique()
        counts = []
        for age, count in age_count.items():
            counts += [age] * count
        plt.hist(counts, bins=bins, weights=np.ones(len(counts)) / len(counts))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    
    @staticmethod
    def plot_age_dist_bar(age_prop):
        labels = list(map(lambda x: f'{x}0-{x}9', age_prop.index))
        plt.bar(x=age_prop.index, height=age_prop.values, width=0.8, tick_label=labels)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    @staticmethod
    def list_individual_kw_counts(df):
        kw_counts = df.groupby('Content')['DocLabel'].nunique()
        doc_count = df.groupby('DocLabel').ngroups
        kw_prop = kw_counts / doc_count
        kw_prop.sort_values(ascending=False, inplace=True)

        return pd.DataFrame(data={'count': kw_prop, 'cumsum': kw_prop.cumsum()})

    # Functions for converting Chinese chief complaints to TTAS_codes
    @staticmethod
    def simplify_cc(row):
        simplified_cc = [cc.split()[0] for cc in filter(None, row['CC'].split(';'))]
        deduplicated_cc = list(set(simplified_cc))
        row['CC'] = ';'.join(deduplicated_cc)

        return row  

    def cc_to_ttas(self, row):
        with open(self.ttas_path, mode="rt", encoding="utf-8") as f:
            ttas_dict = json.load(f)

        cc_list = []
        for cc in row['CC'].split(';'):
            try:
                if (ttas_dict[cc] not in cc_list):
                    cc_list.append(ttas_dict[cc])
            except KeyError:
                cc_list.append(cc)

        row['CC'] = ';'.join(cc_list)
        return row
    
    def build_kw_subdf_rel(self, prop: int, sub_df: pd.DataFrame, ref_df: pd.DataFrame, target: str) -> pd.DataFrame:
        if target not in ["cc", "icd"]:
            ValueError("target must be 'cc' or 'icd'")
        # Get keyword list of keywords which appear in more than [prop] of documents
        doc_count = sub_df.groupby("DocLabel").ngroups
        kw_counts = sub_df.groupby("Content")["DocLabel"].nunique().sort_values(ascending=False)
        kws = kw_counts[kw_counts >= prop * doc_count]
        # Construct DataFrame
        reldf = pd.DataFrame(index=kws.index, columns=["label", "freq", "fisher", "cTF", "IDF", "cTF-IDF", f"{target}_related"], dtype=np.float64)
        reldf["freq"] = kw_counts / doc_count # frequency
        # fill the values keyword by keyword
        kw2label = self.map_kw_to_cat()
        for kw in kws.index: # label, fisher, cTF, IDF, cTF-IDF, cc/icd_related
            reldf.at[kw, "label"] = kw2label.get(kw)
            reldf.at[kw, "fisher"] = self.stats.test_kw_rel(kw, sub_df, ref_df.drop(sub_df.index))[1]
            ctf_idf_results = self.stats.c_tf_idf(kw, sub_df, ref_df)
            for col, result in zip(["cTF-IDF", "cTF", "IDF"], ctf_idf_results):
                reldf.at[kw, col] = result
            reldf.at[kw, f"{target}_related"] = self.stats.related_kw_prop(kw, sub_df, target)
        # Type conversion
        return reldf.astype({"label": np.int8})

def main():
    # test your code here

    return 0

if __name__ == '__main__':
    main()