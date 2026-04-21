import nbformat

notebook_path = '/Users/jayden/Desktop/7012 datamining and text/project_march/ARIN7102_Group_Project/app/interactions/data_collection_eda.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

new_source = """# Visualize: how many unique columns per category + which datasets contribute
raw_cleaned_pairs = [
    ('DS1', 'drug-prescription-to-disease-dataset', 'final.csv', 'final_cleaned.csv'),
    ('DS2', 'drugs-side-effects-and-medical-condition', 'drugs_side_effects_drugs_com.csv', 'drugs_side_effects_drugs_com_cleaned.csv'),
    ('DS3', 'drugs-related-to-common-treatments', 'drugs_for_common_treatments.csv', 'drugs_for_common_treatments_cleaned.csv'),
    ('DS4', 'kuc-hackathon-winter-2018', 'drugsComTrain_raw.csv', 'drugsComTrain_raw_cleaned.csv'),
    ('DS5', 'disease-symptom-description-dataset', 'dataset.csv', 'dataset_cleaned.csv'),
    ('DS6', 'disease-symptom-description-dataset', 'symptom_Description.csv', 'symptom_Description_cleaned.csv'),
    ('DS7', 'disease-symptoms-and-patient-profile-dataset', 'Disease_symptom_and_patient_profile_dataset.csv', 'Disease_symptom_and_patient_profile_dataset_cleaned.csv'),
    ('DS8', 'symptom2disease', 'Symptom2Disease.csv', 'Symptom2Disease_cleaned.csv'),
]

# 1. Column count per category
fig1, ax1 = plt.subplots(figsize=(8, 5))
cat_counts = col_inv_df.groupby('Category')['Column'].nunique().sort_values()
cat_counts.plot(kind='barh', ax=ax1, color='#4C72B0')
ax1.set_xlabel('Unique Column Count')
ax1.set_title('Column Count by Semantic Category')
plt.tight_layout()
plt.show()

# 2. Heatmap — which dataset contributes to which category
fig2, ax2 = plt.subplots(figsize=(8, 5))

source_names = [name.split(': ')[1] if ': ' in name else name for name in datasets.keys()]
cat_names = sorted(col_inv_df['Category'].unique())
heat = pd.DataFrame(0, index=cat_names, columns=source_names)
for _, row in col_inv_df.iterrows():
    heat.loc[row['Category'], row['Source']] += 1

# Map dataset IDs from raw_cleaned_pairs for the labels
heatmap_labels = []
ds_ids = [pair[0] for pair in raw_cleaned_pairs]
for i, src in enumerate(source_names):
    if i < len(ds_ids):
        heatmap_labels.append(ds_ids[i])
    else:
        heatmap_labels.append(src)

im = ax2.imshow(heat.values, cmap='Blues', aspect='auto')
ax2.set_xticks(range(len(heatmap_labels)))
ax2.set_xticklabels(heatmap_labels, rotation=45, ha='right', fontsize=9)
ax2.set_yticks(range(len(cat_names)))
ax2.set_yticklabels(cat_names, fontsize=9)
ax2.set_title('Dataset × Category Contribution')
for i in range(len(cat_names)):
    for j in range(len(source_names)):
        v = heat.values[i, j]
        if v > 0:
            ax2.text(j, i, str(v), ha='center', va='center', fontsize=8, color='white' if v > 3 else 'black')
plt.colorbar(im, ax=ax2, label='# columns')

plt.tight_layout()
plt.show()"""

for cell in nb.cells:
    if cell.cell_type == 'code' and '# Visualize: how many unique columns per category' in cell.source:
        cell.source = new_source
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print('Notebook updated successfully.')
