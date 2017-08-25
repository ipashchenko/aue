import os
import numpy as np
import pandas as pd
from data import remove_correlated_features
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from ae_transform import AETransform
import matplotlib.pyplot as plt


target = 'variable'
data_dir = '/home/ilya/github/ogle'
df_vars = pd.read_pickle(os.path.join(data_dir, "features_vars.pkl"))
df_vars[target] = 1
df_const = pd.read_pickle(os.path.join(data_dir, "features_const.pkl"))
df_const[target] = 0
# df = pd.concat((df_vars, df_const), ignore_index=True)
# df = df.loc[:, ~df.columns.duplicated()]
df_const = remove_correlated_features(df_const, r=0.95)
df_const = df_const.drop(target, axis=1)
df_const = df_const.loc[:, ~df_const.columns.duplicated()]
features_names = list(df_const.columns)
df_vars = df_vars[features_names]
df_vars = df_vars.loc[:, ~df_vars.columns.duplicated()]

X_const = df_const.values
X_const = MinMaxScaler().fit_transform(X_const)
X_const_train, X_const_test = train_test_split(X_const, test_size=0.25)

aet = AETransform(30, dims=[120, 60], loss='mean_squared_error')
X_const_train_ = aet.fit_transform(X_const_train, validation_split=0.25,
                                   epochs=100, batch_size=1024)
aet.summary()
aet.plot_history(save_file="ae_training_.png")


mse_const = np.mean(np.power(X_const_train - X_const_train_, 2), axis=1)

X_vars = df_vars.values
X_vars = MinMaxScaler().fit_transform(X_vars)

X_vars_ = aet.transform(X_vars)
mse_vars = np.mean(np.power(X_vars - X_vars_, 2), axis=1)

X_const_test_ = aet.transform(X_const_test)
mse_const_test = np.mean(np.power(X_const_test - X_const_test_, 2), axis=1)


fig, axes = plt.subplots(1, 1)
axes.hist(mse_const, color='green', alpha=0.3, range=[0, 0.1], bins=100,
          label="Const - used in training", normed=True)
axes.hist(mse_const_test, color='blue', alpha=0.3, range=[0, 0.1], bins=100,
          label="Const - unseen", normed=True)
axes.hist(mse_vars, color='red', alpha=0.3, range=[0, 0.1], bins=100,
          label="Variables - unseen", normed=True)
axes.set_ylabel("Number of light curves")
axes.set_xlabel("Reconstruction MSE")
axes.legend(loc="best")
fig.tight_layout()
fig.savefig("AE_result_normed_.png", dpi=300)

fig, axes = plt.subplots(1, 1)
axes.hist(mse_const, color='green', alpha=0.3, range=[0, 0.1], bins=100,
          label="Const - used in training")
axes.hist(mse_const_test, color='blue', alpha=0.3, range=[0, 0.1], bins=100,
          label="Const - unseen")
axes.hist(mse_vars, color='red', alpha=0.3, range=[0, 0.1], bins=100,
          label="Variables - unseen")
axes.set_ylabel("Number of light curves")
axes.set_xlabel("Reconstruction MSE")
axes.legend(loc="best")
fig.tight_layout()
fig.savefig("AE_result_.png", dpi=300)

