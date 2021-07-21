from sklearn.utils import resample

import pandas as pd

    def df_sampling(df: pd.DataFrame, n = 1000, up_sample = True):
        ''' resampling of dataset'''

        assert ('Class' in df.columns) == True

        minority_class = df['Class'].value_counts(ascending=True).index[0]
        assert minority_class == 1
        
        if up_sample:
            df_to_sample = df[df['Class']==minority_class]
            df_to_add = df[df['Class']!=minority_class]
        else:
            df_to_sample = df[df['Class']!=minority_class]
            df_to_add = df[df['Class']==minority_class]

        df_result = resample(df_to_sample, n_samples=n, replace = True, random_state=42)

        return pd.concat([df_result,df_to_add], axis=0)


if __name__ == "__main__":

    df = pd.read_csv('creditcard.csv')
    r = df_sampling(df, 1000)
    print(r.shape)
