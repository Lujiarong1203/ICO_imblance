import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

df = pd.read_csv('data/dataset_fraud_14.11_2020.csv', sep=";")
# df = pd.read_csv("dataset_fraud_14.11_2020.csv", sep=";")
print(df)

# Viewing Label Ratio
index = df.fraud
print(index.value_counts())
print("percentage of fraud cases in the dataset: "+str(224 / 6040))
print(str(1-(224/6040)))

# Viewing Feature Counts
index = df.type
print(index.value_counts())

index = df.airdrop
print(index.value_counts())

index = df.agent
print(index.value_counts())

index = df.rating_overall
print(index.value_counts())

index = df.bounty
print(index.value_counts())

index = df.kyc_status_all
print(index.value_counts())

index = df.restrictions
print(index.value_counts())

# Processing social media data
df['github']=df['sm_gh'].str.contains("http", na=False).astype(float)
df['facebook']=df['sm_fb'].str.contains("http", na=False).astype(float)
df['twitter']=df['sm_tw'].str.contains("http", na=False).astype(float)
df['telegram']=df['sm_tg'].str.contains("http", na=False).astype(float)
df['youtube']=df['sm_yt'].str.contains("http", na=False).astype(float)
df['bitcointalk']=df['sm_bct'].str.contains("http", na=False).astype(float)
df['instagram']=df['sm_ig'].str.contains("http", na=False).astype(float)
df['linkedin']=df['sm_li'].str.contains("http", na=False).astype(float)
df['medium']=df['sm_med'].str.contains("http", na=False).astype(float)
df['reddit']=df['sm_red'].str.contains("http", na=False).astype(float)

# Process white papers and other information
df['Whitepaper']=(df['whitepaper'].str.len() > 1).astype(float)
df['About']=(df['about'].str.len() > 1).astype(float)
df['Introduction']=(df['intro'].str.len()>1).astype(float)
df['Agent']=df['agent'].str.contains("Yes", na=False).astype(float)
df['Airdrop']=df['airdrop'].str.contains("Yes", na=False).astype(float)
df['Add_token']=df['add_token'].str.contains("Yes", na=False).astype(float)
df['Whitelist']=(df['whitelist'].str.len()>3).astype(float)

bountykeywords = ['Yes', 'Available']
df['Bounty']=df['bounty'].str.contains('|'.join(bountykeywords), na=False).astype(float)

df['KYC']=df['kyc_status_all'].str.contains("passed", na=False).astype(float)

premiumkeywords = ['Premium', 'verified_user']
df['Premium']=df['premium'].str.contains('|'.join(premiumkeywords), na=False).astype(float)

regkeywords = ['KYC & Whitelist', 'KYC', 'Whitelist', 'Yes']
df['Registration']=df['registration'].str.contains('|'.join(regkeywords), na=False).astype(float)

# Extract the number in rat_count
df['rating_count']=df['rat_count'].apply(lambda x:
                                         int(x.split(' ')[0]) if x is not np.NaN
                                         else x)

index = df.accepting
print(index.value_counts())

Acckeywords = ['BTC', 'ETH']
df['Accepting']=df['accepting'].str.contains('|'.join(Acckeywords), na=False).astype(float)

index = df.platform
print(index.value_counts())

df['Platform']=df['platform'].str.contains("Ethereum", na=False).astype(float)

index = df.kyc_num
print(index.value_counts())

df['KYC_num']=df['kyc_num'].apply(lambda x:
                                  int(x.split(' ')[0]) if x != 'Not conducted yet' and x is not np.NaN
                                  else 0)

df['Category_sum']=df['category'].apply(lambda x:
                                    len(x.split(';')) if x is not np.NaN
                                    else x)
print(df['Category_sum'])
print(df.shape)


# Take all 224 fraud samples
df.sort_values(by=['fraud'], inplace=True)
df_fraud = df.tail(224)
print(df_fraud.shape)

# Filtered values are passed kyc, and the values of feature kyc_status_all in the data are Passed.
# The total number of values is 1345.
df_kyc = df[df.kyc_status_all.eq('passed')]
df_kyc.sort_values(by=['fraud'],inplace=True)
print(df_kyc.shape)
index=df_kyc.fraud
print('After filter the data that contains passed：', index.value_counts())

df_balanced = pd.concat([df_kyc.head(1323), df_fraud])
df_balanced.reset_index()
print(df_balanced.shape)
index = df_balanced.fraud
print('new data：', index.value_counts())

# Extract partial features
data_1 = df_balanced[['github','facebook','twitter','telegram','youtube',
                      'bitcointalk','instagram','linkedin','medium','reddit',
                      'Whitepaper','About','rating_count','rating_overall','Bounty',
                      'KYC', 'Registration', 'Premium', 'Introduction', 'Category_sum',
                      'Accepting', 'Platform', 'rating_ai', 'KYC_num', 'Whitelist',
                      'fraud']]
print(data_1.shape)

print(data_1.isnull().sum())

index = data_1.KYC
print('kyc count:', index.value_counts())

print(data_1.dtypes)

data_1.to_csv(path_or_buf=r'data/data_1.csv', index=None)