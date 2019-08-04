import pandas as pd 
import numpy as np 



df = pd.read_csv('spam.csv', encoding='latin')


print(df.head())


df_spam = df[df['v1'] == 'ham']




df_spam = df_spam.dropna(axis=1)

df_spam = df_spam[['v2']]

print(type(df_spam.values))

# np.chararray.encode('utf-8')

np.savetxt('nonspam_data.txt', df_spam.values, fmt='%s', encoding='utf-8')


# df_spam.to_csv('spam_data.csv', columns=['v2'], sep='\n')


# df_ham = 