import pandas as pd
import re

# Let's assume the file content is loaded into the DataFrame
# For the purpose of this example, we'll create a DataFrame with similar patterns as seen in the image

# Sample data mimicking the user's data structure based on the image provided
xlsx_data = "c:\\Users\\USER\\Downloads\\VFSS_residue_image_list_김소현 (1).xlsx"
data = pd.read_excel(xlsx_data, engine='openpyxl',
                     sheet_name=1)  # For the second sheet
# Create a DataFrame
df = pd.DataFrame(data)
df.head()

df['Name'] = df['영상파일명']


def extract_name(path):
    # This regex looks for a backslash followed by any combination of letters (Korean included), numbers, and underscores
    # It stops when it finds a space followed by an open parenthesis and a number
    # Make sure path is a string, if not, return None
    if not isinstance(path, str):
        return None
    # This regex looks for a backslash followed by any combination of letters (Korean included), numbers, and underscores
    # It stops when it finds a space followed by an open parenthesis and a number
    match = re.search(r'\\([가-힣A-Za-z0-9_]+)\s+\(\d+\)', path)
    return match.group(1) if match else None


# Apply the function to extract names
df['Name'] = df['영상파일명'].apply(extract_name)

# Now, let's get the distinct names
distinct_names = df['Name'].unique()
print(distinct_names)

# Return the distinct names and their count
print(len(distinct_names))
