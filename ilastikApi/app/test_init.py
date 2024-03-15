import openpyxl
import json
import re

def calculate_status(cell_formula):
    # Implement your logic to calculate the status based on the formula
    # For now, this example just extracts 'Y' or 'N' from the formula
    result = re.search(r'["\'](Y|N)["\']', cell_formula)
    return result.group(1) if result else ""
def build_path(parent_path, child):
    separator = "->" if parent_path else ""
    return f"{parent_path}{separator}{child}"

def extract_data(filename):
    workbook = openpyxl.load_workbook(filename)
    sheet = workbook.active

    data = []
    current_path = ""
    current_parents = [None, None, None, None]

    for row in sheet.iter_rows(min_row=8,max_row=30):
        cell_a = row[0].value
        cell_b = row[1].value if row[1].value else ""

        level = 0

        if cell_a:
            if '    ' in cell_a:
                level = 3
            else:
                level = 2
            if cell_a.strip():
                if row[0].fill.fgColor.rgb == "FF1F4A7F":  # Dark blue: node
                    level = 0
                elif row[0].fill.fgColor.rgb == "FF80ABE0":  # Light blue: child
                    level = 1

        if cell_a is None and cell_b:
            level = 4

        child_name = cell_a.strip() if cell_a else cell_b.strip()
        status = row[2].value if row[2].value else ""
        comment = row[3].value if row[3].value else ""

        # Calculate status if it's not in the specified values
        if status not in ['Tbl', 'Y', 'P', 'D', 'N']:
            # Assuming status is a formula in this case
            calculated_status = calculate_status(status)
        else:
            calculated_status = status
        child_data = {
            "name": child_name,
            "status": calculated_status,
            "comment": comment,
            "path": build_path(current_path, child_name),
            "children": []
        }
        print("this is current path before",child_data,level)
        print("this is current path after",current_path)
        if level == 1:
            child_data["path"] = build_path("", child_name)
            data.append(child_data)
            current_parents[0] = child_data
        elif level == 2:
            child_data["path"] = build_path(current_parents[0]["path"], child_name)
            current_parents[0]["children"].append(child_data)
            current_parents[1] = child_data
        elif level == 3:
            child_data["path"] = build_path(current_parents[1]["path"], child_name)
            current_parents[1]["children"].append(child_data)
            current_parents[2] = child_data
        elif level == 4:
            child_data["path"] = build_path(current_parents[2]["path"], child_name)
            current_parents[2]["children"].append(child_data)

    return data

# Example usage
data = extract_data('TIWS CFL.xlsx')
json_data = json.dumps(data, indent=4)

with open('output.json', 'w') as json_file:
    json_file.write(json_data)
