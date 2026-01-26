import json

notebook_path = 'K_means_Clustering.ipynb'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_data = json.load(f)

    cells = nb_data['cells']
    fixed = False
    
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if "personas = df.groupby('Cluster').agg" in source and "'Genre'" in source:
                print(f"Found problematic cell: {cell.get('id', 'unknown ID')}")
                new_source = []
                for line in cell['source']:
                    # Replace 'Genre' with 'Gender' in the aggregation dictionary
                    new_line = line.replace("'Genre'", "'Gender'")
                    new_source.append(new_line)
                cell['source'] = new_source
                fixed = True
                print("Applied fix: Replaced 'Genre' with 'Gender'")
                break
    
    if fixed:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb_data, f, indent=1)
        print("Successfully saved changes to notebook.")
    else:
        print("Could not find the problematic cell or it makes no reference to 'Genre'.")

except Exception as e:
    print(f"Error: {e}")
