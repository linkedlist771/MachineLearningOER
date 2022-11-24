import plotly.figure_factory as ff
import numpy as np
a, b = np.mgrid[0:1:30j, 0:1:20j]
mask = a + b <= 1
a, b = a[mask], b[mask]
c = 1-a-b
d = 0*c
ternay_Data = pd.DataFrame({"Ce":a, "Fe":b, "Ni":c, "Co": d})
df = ternay_Data.copy()
df["formula"] = 1
df
for index, row in df.iterrows():
    ni_fraction = row["Ce"]
    fe_fraction = row["Ni"]
    co_fraction = row["Fe"]
    ce_fraction = row["Co"]
    conte = f"Ce{ni_fraction}Ni{fe_fraction}Fe{co_fraction}Co{ce_fraction}"
    row.loc["formula"] = conte
    df.iloc[index] = row
    #print() # 输出每行的索引值
df.head()
from matminer.featurizers.conversions import StrToFemposition
df = StrToFemposition().featurize_dataframe(df, "formula")
df.head()
from matminer.featurizers.composition import ElementProperty
ep_feat = ElementProperty.from_preset(preset_name="magpie")
df = ep_feat.featurize_dataframe(df, col_id="composition")  # input the "composition" column to the featurizer
df.head()
df_mole_OP = df.drop(['Ce', 'Ni', 'Fe', 'Co'], axis=1)
e = np.array(predictor.predict(df_mole_OP).values)
for color in ['Picnic']:#['Blackbody','Bluered','Blues','Cividis',
    #'Earth','Electric','Greens','Greys','Hot','Jet','Picnic','Portland','Rainbow','RdBu','Reds','Viridis','YlGnBu','YlOrRd']:
    fig = ff.create_ternary_contour(np.array([b, c, a]), e,
                                    pole_labels=[ 'Fe', 'Ni','Ce'],
                                    interp_mode='cartesian',
                                    ncontours=10,
                                    showscale=True,
                                    colorscale=color ,
                                    title='Predicted overpotential(mv)',
                                )
    fig.show()