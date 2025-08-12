networks = ['ConvNeXt', 'ViTB']

import matplotlib

font = {'size': 15}

matplotlib.rc('font', **font)

import pandas as pd


# Read the CSV files into DataFrames
def compute_direct_compliance(file_name_similarity, file_name_confusion, printing=False):
    df1 = pd.read_csv(file_name_similarity)  # Replace 'file1.csv' with your first file path
    df2 = pd.read_csv(file_name_confusion)  # Replace 'file2.csv' with your second file path

    # Create sets of frozensets for node1 and node2 tuples
    set1 = set(df1.apply(lambda row: frozenset([row['node1'], row['node2']]), axis=1))
    set2 = set(df2.apply(lambda row: frozenset([row['node1'], row['node2']]), axis=1))

    # Find the intersection and unique elements
    common_tuples = set1.intersection(set2)
    unique_to_df1 = set1 - set2
    unique_to_df2 = set2 - set1

    # Print results
    print(f"Ratio of Similarities causing direct high confusion: {len(common_tuples) / len(set1):.2f}")
    print(f"Ration of explained high confusion pairs via similarity: {len(common_tuples) / len(set2):.2f}")
    print(f"Number of matching tuples: {len(common_tuples)}")
    print(f"All similarity tuples: {len(set1)}")
    print(f"All confusion tuples: {len(set2)}")

    simi = len(common_tuples) / len(set1)
    conf = len(common_tuples) / len(set2)
    return simi, conf

    if printing:
        print("\nCommon tuples:")
        for t in common_tuples:
            print(t)

        print("\nTuples unique to Representation:")
        for t in unique_to_df1:
            print(t)

        print("\nTuples unique to Confusion:")
        for t in unique_to_df2:
            print(t)
#%%
sims = []
confs = []
for model_name in networks:
    dir_name1 = f"RESULTS_ROOT/DF/NCSM/200/{model_name}/results.csv"
    dir_name2 = f"RESULTS_ROOT/DF/CCSM/200/{model_name}/results.csv"
    print(model_name)
    sim, conf = compute_direct_compliance(dir_name1, dir_name2)
    sims.append(sim)
    confs.append(conf)

df = pd.read_csv("RESULTS_ROOT/results_epoch_200.csv")
#%%
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text



# Create the scatter plot
plt.figure(figsize=(8, 4))
sns.scatterplot(x=df.NCSM_depth, y=confs, s=100, color="blue", edgecolor="black")

# Add labels to each point
for i, label in enumerate(networks):
    plt.text(df.NCSM_depth[i] + 0.01, confs[i], label, fontsize=15, ha='left', va='center', color='black')


plt.xlabel("Depth", fontsize=15)
plt.ylabel("Confusions explained", fontsize=15)
plt.grid(True, linestyle='--', alpha=0.6)

# Display the plot
plt.tight_layout()
plt.show()


# Create the scatter plot
plt.figure(figsize=(8, 4))
sns.scatterplot(x=df.NCSM_depth, y=sims, s=100, color="blue", edgecolor="black")

# Add labels to each point
for i, label in enumerate(networks):
    plt.text(df.NCSM_depth[i] + 0.01, sims[i], label, fontsize=15, ha='left', va='center', color='black')


plt.xlabel("Depth", fontsize=15)
plt.ylabel("Similarities causing confusion", fontsize=15)
plt.grid(True, linestyle='--', alpha=0.6)

# Display the plot
plt.tight_layout()
plt.show()


# Create the scatter plot
plt.figure(figsize=(8, 4))
sns.scatterplot(x=df.NCSM_depth, y=df.accuracy, s=100, color="blue", edgecolor="black")

# Add labels to each point
texts = []
for i, label in enumerate(networks):
    texts.append(plt.text(df.NCSM_depth[i], df.accuracy[i], label, fontsize=15, ha='left', va='center', color='black'))

# Adjust text to avoid overlaps
adjust_text(
    texts,
    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)  # Optional: Add arrows for clarity
)

plt.xlabel("Depth", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Create the scatter plot
plt.figure(figsize=(8, 4))
sns.scatterplot(x=df.NCSM_depth, y=df.losses, s=100, color="blue", edgecolor="black")

# Add labels to each point
texts = []
for i, label in enumerate(networks):
    texts.append(plt.text(df.NCSM_depth[i], df.losses[i], label, fontsize=15, ha='left', va='center', color='black'))

# Adjust text to avoid overlaps
adjust_text(
    texts,
    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)  # Optional: Add arrows for clarity
)

plt.xlabel("Depth", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.grid(True, linestyle='--', alpha=0.6)

# Display the plot
plt.show()
#%%
import pandas as pd
from itertools import combinations


def compute_extended_compliance(file_name_similarity, file_name_confusion, printing=False):
    df1 = pd.read_csv(file_name_similarity)  # Replace 'file1.csv' with your first file path
    df2 = pd.read_csv(file_name_confusion)  # Replace 'file2.csv' with your second file path

    # Function to generate all possible tuples within a group in a DataFrame
    def generate_all_tuples(group):
        nodes = set(group['node1']).union(set(group['node2']))  # Collect all unique nodes in the group
        all_tuples = {frozenset(pair) for pair in combinations(nodes, 2)}  # Create all combinations of 2 nodes
        return all_tuples

    # Function to check if tuples from source_df exist in any group of target_df
    def check_tuples_in_groups(source_df, target_df):
        matches = []
        unmatched = []
        for _, source_row in source_df.iterrows():
            # Create a tuple from the current row
            tuple_from_source = frozenset([source_row['node1'], source_row['node2']])

            found = False
            # Check each group in the target DataFrame
            for group_id in target_df['ID'].unique():
                group = target_df[target_df['ID'] == group_id]
                target_tuples = generate_all_tuples(group)  # Generate all possible tuples for the group

                # Check if the tuple exists in the target group's tuples
                if tuple_from_source in target_tuples:
                    matches.append((tuple_from_source, group_id))
                    found = True
                    break  # No need to check further groups for this tuple

            if not found:
                unmatched.append(tuple_from_source)

        return matches, unmatched

    # Check for tuples from df1 in groups of df2
    matches_for_df1, unmatched_for_df1 = check_tuples_in_groups(df1, df2)

    # Check for tuples from df2 in groups of df1
    matches_for_df2, unmatched_for_df2 = check_tuples_in_groups(df2, df1)
    if printing:
        print("\nSimilarities found in Confusion groups:")
        for t, group_id in matches_for_df1:
            print(f"Tuple: {t}, Found in Group ID: {group_id}")
    print(
        f"Ratio of similarities causing confusion to all similarities: {len(matches_for_df1) / (len(matches_for_df1) + len(unmatched_for_df1)):.2f}")

    if printing:
        print("\nConfusions found in Similarity Groups:")
        for t, group_id in matches_for_df2:
            print(f"Tuple: {t}, Found in Group ID: {group_id}")
    print(
        f"Ratio of confusions explained via similarity: {len(matches_for_df2) / (len(matches_for_df2) + len(unmatched_for_df2)):.2f}")

    if printing:
        print("\nSimilarities not found in Confusion groups:")
        for t in unmatched_for_df1:
            print(t)
        print(
            f"Ratio of similarities NOT causing confusion to all similarities: {len(unmatched_for_df1) / (len(matches_for_df1) + len(unmatched_for_df1)):.2f}")

        print("\nConfusions not found in Similarity Groups:")
        for t in unmatched_for_df2:
            print(t)
        print(
            f"Ratio of confusions NOT explained via similarity: {len(unmatched_for_df2) / (len(matches_for_df2) + len(unmatched_for_df2)):.2f}\n")
    sims = len(matches_for_df1) / (len(matches_for_df1) + len(unmatched_for_df1))
    confs = len(matches_for_df2) / (len(matches_for_df2) + len(unmatched_for_df2))
    return sims, confs

sims = []
confs = []
for model_name in networks:
    dir_name1 = f"RESULTS_ROOT/DF/NCSM/200/{model_name}/results.csv"
    dir_name2 = f"RESULTS_ROOT/DF/CCSM/200/{model_name}/results.csv"
    print(model_name)
    sim, conf = compute_extended_compliance(dir_name1, dir_name2, printing=False)
    sims.append(sim)
    confs.append(conf)
#%%
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text



# Create the scatter plot
plt.figure(figsize=(8, 4))
sns.scatterplot(x=df.NCSM_depth, y=confs, s=100, color="blue", edgecolor="black")

# Add labels to each point
for i, label in enumerate(networks):
    plt.text(df.NCSM_depth[i] + 0.01, confs[i], label, fontsize=15, ha='left', va='center', color='black')


plt.xlabel("Depth", fontsize=15)
plt.ylabel("Confusions explained \n (extended neighborhood)", fontsize=15)
plt.grid(True, linestyle='--', alpha=0.6)

# Display the plot
plt.tight_layout()
plt.show()


# Create the scatter plot
plt.figure(figsize=(8, 4))
sns.scatterplot(x=df.NCSM_depth, y=sims, s=100, color="blue", edgecolor="black")

# Add labels to each point
for i, label in enumerate(networks):
    plt.text(df.NCSM_depth[i] + 0.01, sims[i], label, fontsize=15, ha='left', va='center', color='black')


plt.xlabel("Depth", fontsize=15)
plt.ylabel("Similarities causing confusion \n (extended neighborhood)", fontsize=15)
plt.grid(True, linestyle='--', alpha=0.6)

# Display the plot
plt.tight_layout()
plt.show()
#%%
