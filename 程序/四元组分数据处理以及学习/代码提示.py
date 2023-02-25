predictor_leaderboard.plot.bar(figsize=(20,12))

# make some white space of the bottom and no white space on the top
# when save the fig
plt.subplots_adjust(bottom=0.2, top=0.9)
plt.xticks(fontsize=20, rotation=0)
plt.xticks(fontsize=20)
plt.xlabel('predictor',fontsize=20)
plt.ylabel('score',fontsize=20)
plt.legend(fontsize=20)
plt.savefig("特征抽取后的各个模型的结果图", dpi=300)