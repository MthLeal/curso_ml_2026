#%%
import pandas as pd

clones = pd.read_parquet("data/dados_clones.parquet")
clones.head()

#%%
target = 'Status '

features = ['Massa(em kilos)',
            'General Jedi encarregado',
            'Estatura(cm)',
            'Distância Ombro a ombro',
            'Tamanho do crânio',
            'Tamanho dos pés',
            'Tempo de existência(em meses)']

y = clones[target]
X = clones[features]
#%%
X['Tamanho dos pés'].drop_duplicates()
#%%

X = X.replace({
    'Yoda':1, 'Shaak Ti':2,'Obi-Wan Kenobi':3,'Aayla Secura':4,'Mace Windu':5,
    'Tipo 1':1,
    'Tipo 2':2,
    'Tipo 3':3,
    'Tipo 4':4,
    'Tipo 5':5,
})
#%%
from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(X=X, y=y)

#%%
import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model, feature_names=features,
               class_names=model.classes_,
               filled=True, max_depth=4)
# %%
