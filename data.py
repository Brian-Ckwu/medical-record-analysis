import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import json

class Data:
    def __init__(self, data_path, icd_to_dx_path="./icd_to_dx/allcodes.json", ttas_path="./ttas/ttas.json"):
        # Convert the csv file to DataFrame
        self.__df = pd.read_csv(data_path, index_col='index', encoding='utf-8', dtype={'ICD9': np.str_})
        # Fill the empty values in the 'posOrNeg' column with 3.0
        self.__df['posOrNeg'].fillna(3.0, inplace=True)
        # Keyword classification
        self.labels = {
            4: 'diagnosis',
            5: 'drug',
            6: 's/s',
            7: 'surgery',
            8: 'others',
            11: 'non_surgery'
        }
        # Classify the icdcodes as symptom_dx or disease_dx
        self.s_df = self.__df.loc[(self.__df['ICD9'] >= '780') & (self.__df['ICD9'] < '800')] # DataFrame of symptom_dx
        self.d_df = self.__df.loc[(self.__df['ICD9'] < '780') | (self.__df['ICD9'] >= '800')] # DataFrame of disease_dx
        # ICD-9 code to diagnosis conversion file
        with open(icd_to_dx_path, 'rt') as f:
            self.icd_to_dx = json.load(f)
        # 'TTAS code to name' conversion file
        self.ttas_path = ttas_path
        with open(ttas_path, mode='rt', encoding='utf-8') as f:
            name_to_code = json.load(f)
            code_to_name = {}
            for name, code in name_to_code.items():
                code_to_name[code] = name
            self.ttas_dict = code_to_name
    
    # Get the DataFrame
    def get_df(self):
        return self.__df

    # Get DataFrame from TTAS code
    def get_df_from_cc(self, ttas_code):
        return self.__df[self.__df['CC'].str.contains(ttas_code)]

    # Get the number of medical records (ssd / dsd / total)
    def get_doc_counts(self):
        def get_doc_count(df):
            return df.groupby('DocLabel').ngroups
            
        return {
            'SSD': get_doc_count(self.s_df),
            'DSD': get_doc_count(self.d_df),
            'All': get_doc_count(self.__df)
        }
    
    # Get doc counts stratified by age
    def get_age_doc_counts(self):
        def get_age_doc(df):
            return df.groupby('Age')['DocLabel'].nunique().groupby(lambda age: age // 10).sum()

        d = {}
        for name, df in zip(['SSD', 'DSD', 'All'], [self.s_df, self.d_df, self.get_df()]):
            doc_count = get_age_doc(df)
            doc_count.at[8] = doc_count[doc_count.index >= 8].sum()
            doc_count.drop([9, 10], inplace=True)
            d[name] = doc_count.values

        return pd.DataFrame(data=d, index=[f'{i}0-{i}9' for i in range(2, 8)] + ['80+'])
    
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
        for dx_type, df in zip(['SSD', 'DSD'], [self.s_df, self.d_df]):
            cc_doc_counts[dx_type] = get_cc_doc_counts_dict(df)
        # Construct DataFrame
        df = pd.DataFrame(data=cc_doc_counts).fillna(0)
        df['All'] = df['SSD'] + df['DSD']
        df = df.sort_values(by='All', ascending=False).head(num)
        ttas_names = [self.ttas_dict[code] for code in df.index]
        df['Names'] = ttas_names

        return df[['Names', 'SSD', 'DSD', 'All']].convert_dtypes()
    
    # Get all icdcodes
    def get_icdcodes(self):
        return self.__df['ICD9'].unique()
    
    # Get ndoc (number of medical records) of an icdcode
    def get_icdcode_ndoc(self, icdcode):
        return self.get_icdcodes_ndoc()[icdcode]
    
    def get_icdcodes_ndoc(self):
        return self.__df.groupby('ICD9')['DocLabel'].nunique()
    
    # Get the icdcodes of which ndoc is greater than a certain number
    def get_icdcodes_ndoc_gt(self, ndoc):
        icdcodes_ndoc = self.get_icdcodes_ndoc()
        return icdcodes_ndoc[icdcodes_ndoc > ndoc].index.tolist()

    def get_keywords(self):
        return self.__df['Content'].unique()

    def get_keyword_count(self, keyword):  # If a keyword is present in a doc, count once
        keywords_count = self.get_keywords_count()
        count = keywords_count.at[keyword]
        return count

    def get_keywords_count(self):
        return self.__df.groupby('Content')['DocLabel'].nunique()
    
    def get_keywords_count_gt(self, count):
        keywords_count = self.get_keywords_count()
        return keywords_count[keywords_count > count].index.tolist()
    
    def get_keywords_count_in_icdcode(self, icdcode):
        icdcode_df = self.__df[self.__df['ICD9'] == icdcode]
        return icdcode_df.groupby('Content')['DocLabel'].nunique()
    
    def get_icd_keyword_count(self, icdcode, keyword):
        icd_df = self.__df[self.__df['ICD9'] == icdcode][['Content', 'posOrNeg', 'DocLabel']]
        # Process the data by groupby()
        keywords_count = icd_df.groupby('Content')['DocLabel'].nunique()
        keywords_count_pn = icd_df.groupby(['Content', 'posOrNeg'])['DocLabel'].nunique()
        def try_get_count(group, target):
            try:
                return group.at[target]
            except KeyError:
                return 0
        count = {
            'pos': try_get_count(keywords_count_pn, (keyword, 1.0)),
            'neg': try_get_count(keywords_count_pn, (keyword, 2.0)),
            'total': try_get_count(keywords_count, keyword)
        }
        return count
    
    def sort_and_save(self, filepath, by='Content'):
        sorted_df = self.__df.sort_values(by, axis='index')
        sorted_df.to_csv(filepath, encoding='big5')
    
    def analyze_keyword_in_icdcode(self, keyword, icdcode):
        total_df = self.__df[(self.__df['ICD9'] == icdcode) & (self.__df['Content'] == keyword)]
        pos_df = total_df[total_df['posOrNeg'] == 1]
        neg_df = total_df[total_df['posOrNeg'] == 2]
        result = {
            'total': total_df['DocLabel'].nunique(),
            'pos': pos_df['DocLabel'].nunique(),
            'neg': neg_df['DocLabel'].nunique()
        }
        return result
    
    # Send the DataFrame you want to describe (e.g. s_df or d_df)
    def describe(self, df):
        # Initiate variables
        desc = {'ndoc': 0, 'kw_stat': {}}
        nkw_data = {'total': [], 'pos': [], 'neg': []}
        for label_name in self.labels.values():
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
            for label, label_name in self.labels.items():
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

        for dx_name, df in zip(['SSD', 'DSD', 'All'], [self.s_df, self.d_df, self.get_df()]):
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
            for value, label in self.labels.items():
                dd[label] = df[df['label'] == value].groupby('DocLabel')['Content'].nunique().sum()
            
            # Construct series
            s = pd.Series(data=dd) / doc_count
            d[dx_name] = s
        
        return pd.DataFrame(data=d)

    def group_icdcodes_ndoc(self):
        return self.__df.groupby('ICD9')['DocLabel'].nunique().groupby(lambda code: str(code).split('.')[0].zfill(3)).sum()

    # Func: convert icdcodes to diagnosis names
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
        for dx_type, df in zip(dx_types, [self.s_df, self.d_df]):
            dx_props[dx_type] = df.groupby('ICD9')['DocLabel'].nunique().groupby(lambda icd: icd[:3]).sum().sort_values(ascending=False).head(num)

        return [pd.DataFrame(data={'names': self.get_dx_names(dx_props[dx_type].index),'counts': dx_props[dx_type]}) for dx_type in dx_types]
    
    # Plot the ndoc of symptom_dx & disease_dx
    def plot_ndoc(self, s_desc, d_desc): # Get s_desc_ & d_desc from self.describe function
        plt.figure('ndoc', figsize=(5, 3))
        plt.title('ndoc')
        plt.bar([1, 1.6], [s_desc['ndoc'], d_desc['ndoc']], width=0.4)
        plt.xticks([1, 1.6], ['symptom_dx', 'disease_dx'])
    
    # Plot the proportion of positive and negative keywords
    def plot_pos_and_neg(self, s_desc, d_desc): # Get s_desc_ & d_desc from self.describe function
        fig, axs = plt.subplots(1, 2, figsize=(5, 3))
        fig.suptitle('pos and neg')
        for i, desc in enumerate([s_desc, d_desc]):
            total, pos, neg = desc['kw_stat']['total']['mean'], desc['kw_stat']['pos']['mean'], desc['kw_stat']['neg']['mean']
            dic = {'pos': pos, 'neg': neg, 'unlabelled': total - pos - neg}
            axs[i].pie(dic.values(), labels=dic.keys(), autopct='%1.2f%%')
            axs[i].set_title('symptom_dx' if i == 0 else 'disease_dx')
    
    # Plot the bar chart of each keyword category (comparing symptom_dx & disease_dx)
    def plot_kw_categories(self, s_desc, d_desc): # Get s_desc_ & d_desc from self.describe function
        plt.figure('kw categories', figsize=(5, 3))
        plt.title('kw categories')
        categories = ['s/s', 'diagnosis', 'drug', 'surgery', 'non_surgery', 'others']
        x = np.arange(len(categories))
        for i, desc in enumerate([s_desc, d_desc]):
            cat = np.array([desc['kw_stat'][cat]['mean'] for cat in categories]) / desc['kw_stat']['total']['mean']
            plt.bar(x + i * 0.3, cat, width=0.3, label='symptom_dx' if i == 0 else 'disease_dx')
        plt.legend()
        plt.xticks(x + 0.15, categories)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    # Plot the keyword number distribution of every category
    def plot_nkw_dist_every_cat(self, s_data, d_data, bins): # Get s_desc_ & d_desc from self.describe function
        # Function for formatting p_value
        def format_p(p_value):
            if (p_value < 0.001):
                return 'p < 0.001'
            elif (p_value < 0.05):
                return 'P < 0.05'
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
                axs[i].text(x=0.65 * axs[i].get_xlim()[1], y=(0.65 - j * 0.10) * axs[i].get_ylim()[1], s=f'{round(np.mean(data[category]), 2):.2f}±{round(np.std(data[category]), 2):.2f}', color=f'C{j}', fontsize=13.5)
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
        fig.suptitle(('total_kw' if cat == 'total' else f'{self.labels[cat]}_kw') + f' ({round(p_min * 100, 1)}% - {round(p_max * 100, 1)}%)', x=0.08, y=0.91, fontweight=600)
        for i, df in enumerate([self.s_df, self.d_df]):
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
        df = self.get_df().copy()
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

    def change_kw(self, json_path):
        with open(json_path, mode="rt", encoding="utf-8") as f:
            kw_groups = json.load(f)

        df = self.get_df(cc=True)

        for old, new in kw_groups.items():
            df.loc[df["Content"] == old, "Content"] = new
            print(f"{old} -> {new}")

        return df

    def group_kw(self, json_path):
        with open(json_path, mode="rt", encoding="utf-8") as f:
            kw_groups = json.load(f)
        
        df = self.get_df(cc=True)

        for ref_kw in kw_groups:
            to_be_grouped = kw_groups[ref_kw]
            for kw in to_be_grouped:
                df.loc[df["Content"] == kw, "Content"] = ref_kw
                print(f"{kw} -> {ref_kw}")
            print("--------------------------------")

        return df
    
    def plot_age_nkw_dist(self, target, category=0, interval=10, ax=None):
        if (interval not in [5, 10]):
            raise Exception('The argument interval can only be 5 or 10.')
        
        if (target not in ['sex', 'diagnosis']):
            raise Exception('The argument target can only be "sex" or "diagnosis".')

        # Extract DataFrame of certain category
        labels = self.labels.copy()
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
            l_df, r_df = self.s_df, self.d_df
            l_cat_df, r_cat_df = cat_df[(cat_df['ICD9'] >= '780') & (cat_df['ICD9'] < '800')], cat_df[(cat_df['ICD9'] < '780') | (cat_df['ICD9'] >= '800')]
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
    
    def split_keywords(self, target, delimiter='/'):
        df = self.__df.copy()
        df.loc[df['Content'] == target, 'Content'] = df.loc[df['Content'] == target, 'Content'].str.split(delimiter)
        return df.explode('Content').reset_index(drop=True)
    
    def get_keywords_diff(self, pos_or_neg, category):
        # Count keywords number in SSD and DSD medical records
        kwn = {}
        for dx_type, df in zip(['SSD', 'DSD'], [self.s_df, self.d_df]):
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
    
    # Perform Fisher's exact test on a keyword's frequency in an icdcode
    def test_kw_icd_rel(self, keyword, icdcode, mode):
        counts = {}
        df = self.__df
        icd_df = df[df['ICD9'].str[:len(icdcode)] == icdcode]
        for key, df in zip(['icd', 'all'], [icd_df, df]):
            all_doc_count = df.groupby('DocLabel').ngroups
            kw_doc_count = df[(df['posOrNeg'] == 1.0) & (df['Content'] == keyword)].groupby('DocLabel').ngroups
            counts[key] = [kw_doc_count, all_doc_count - kw_doc_count]
        return stats.fisher_exact([counts['icd'], counts['all']], alternative=mode) # Odds ratio & p-value

    def test_kw_icds_rel(self, keyword, icdcodes, mode='greater'):
        names = self.get_dx_names(icdcodes)
        p_values = [self.test_kw_icd_rel(keyword, icdcode, mode)[1] for icdcode in icdcodes]
        return pd.DataFrame(data={'names': names, 'p_values': p_values}, index=icdcodes)

    # Perform Fisher's exact test on a keyword's frequency between two DataFrames (df2 is usually reference df)
    def test_kw_rel(self, keyword, df1, df2, mode="greater", test="fisher"):
        counts = []

        for df in [df1, df2]:
            all_doc_count = df.groupby("DocLabel").ngroups
            kw_doc_count = df[df["Content"] == keyword].groupby("DocLabel").ngroups

            counts.append([kw_doc_count, all_doc_count - kw_doc_count])
        
        print(counts)

        if test == "fisher":
            result = stats.fisher_exact(counts, alternative=mode)
        elif test == "chi":
            result = stats.chi2_contingency(counts)
        else:
            ValueError("arg test must be 'fisher' or 'chi'")
        
        return result # Return value: (oddsratio, p_value)

    # Make keyword-cc relation (based on Fisher's exact test) DataFrame
    def make_kw_cc_rel_by_test(self, kw_num, cc_num, test="fisher"):
        # Get keywords
        kw_series = self.get_df().groupby("Content")["DocLabel"].nunique()
        kws = kw_series.sort_values(ascending=False)[:kw_num].index
        # Get ccs
        ccs = self.get_cc_doc_counts(cc_num).index

        # Construct df
        df = pd.DataFrame(index=kws, columns=ccs, dtype=np.float64).fillna(0)

        # Loop through ccs
        for cc in ccs:
            cc_df = self.get_df_from_cc(cc)
            not_cc_df = self.__df.drop(cc_df.index)
            for kw in kws:
                p_value = self.test_kw_rel(kw, cc_df, not_cc_df, test=test)[1]
                df.at[kw, cc] = p_value
                print(f"{kw} in {cc}: {p_value:.2f}")
        
        return df
    
    # Make keyword-cc relation DataFrame (sort according to the ratio of cc_related count / all count of a keyword in a chief complaint)
    def make_kw_cc_rel_labelled(self, cc_code):
        # DataFrame of the chief complaint
        df = self.get_df_from_cc(cc_code)

        # Get total count & cc_related count of the keywords
        kw_total_count = df.groupby("Content")["DocLabel"].nunique()
        kw_ccr_count = df[df["CC_Related"] == True].groupby("Content")["DocLabel"].nunique()

        # Concatenate two series and calculate the ratio
        kw_concat = pd.concat([kw_ccr_count, kw_total_count], axis=1).fillna(0).convert_dtypes()
        kw_concat.columns = ["cc_related", "all"]
        kw_concat["ratio"] = kw_concat["cc_related"] / kw_concat["all"]
        kw_concat.sort_values(by="ratio", ascending=False, inplace=True)

        return kw_concat
    
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

def main():
    filename = "./data/all_data/v3/ALL_Data_List_v3_kwcor_grouped_ttas.csv"
    data = Data(filename)

    kw_num = 1000
    cc_num = 15
    rel_df = data.make_kw_cc_rel_by_test(kw_num, cc_num, test="chi")

    out_filename = f"./keywords/relation/cc/cc_{cc_num}_kw_{kw_num}_chi.csv"
    rel_df.to_csv(out_filename, encoding="utf-8")

    return 0

if __name__ == '__main__':
    main()