import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

dataframe= pd.read_pickle("/Users/siddharthsharma/Desktop/tracking-barbell-exercises/data/interim/01_data_processed.pkl")

set_dataframe= dataframe[dataframe["set"]==1]

plt.plot(set_dataframe["acc_y"])


plt.plot(set_dataframe["acc_y"].reset_index(drop=True))
plt.show()

for label in dataframe["label"].unique():
    subset = dataframe[dataframe["label"]==label]
    fig, ax= plt.subplots()
    plt.plot(set_dataframe["acc_y"].reset_index(drop=True), label = label)
    plt.legend()
    plt.show()
    


for label in dataframe["label"].unique():
    subset = dataframe[dataframe["label"]==label]
    fig, ax= plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label = label)
    plt.legend()
    plt.show()
    
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams ["figure.dpi"] = 100
    
    

category_dataframe= dataframe.query("label == 'squat'").query("participant == 'A'").reset_index()
fig, ax= plt.subplots()
category_dataframe.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
plt.show()

participant_dataframe=dataframe.query("label == 'bench'").sort_values("participant").reset_index() 
fig, ax= plt.subplots()
participant_dataframe.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
plt.show()

label= "squat"
participant= "A"
all_axis_dataframe=dataframe.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index()

fig, ax= plt.subplots()
all_axis_dataframe[["acc_x","acc_y","acc_z"]].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
plt.show()

labels=dataframe["label"].unique()
participants=dataframe["participant"].unique()

for label in labels:
    for participant in participants:
        all_axis_dataframe=dataframe.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index()

        if len(all_axis_dataframe)>0:
            fig, ax= plt.subplots()
            all_axis_dataframe[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax)
            ax.set_ylabel("gyr_y")
            ax.set_xlabel("samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()
            plt.show()
            

            

label= "row"
participant= "A"
combined_plot_dataframe=dataframe.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index(drop=True)

fig, ax= plt.subplots(nrows=2, sharex=True, figsize=(20,10))
combined_plot_dataframe[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
combined_plot_dataframe[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])

ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True, shadow=True)
ax[1].set_xlabel("samples")

for label in labels:
    for participant in participants:
        combined_plot_dataframe=dataframe.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index()
        
        if len(combined_plot_dataframe)>0:
            fig, ax= plt.subplots(nrows=2, sharex=True, figsize=(20,10))
            combined_plot_dataframe[["acc_x","acc_y","acc_z"]].plot(ax=ax[0])
            combined_plot_dataframe[["gyr_x","gyr_y","gyr_z"]].plot(ax=ax[1])

            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].set_xlabel("samples")
            plt.savefig(f"/Users/siddharthsharma/Desktop/tracking-barbell-exercises/reports/figures/{label.title()}({participant}) .png")
            plt.show()