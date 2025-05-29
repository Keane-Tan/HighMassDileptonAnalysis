import json

# Open and load the JSON file
with open('src/lib/histogram/plot_settings_stage1.json', 'r') as f:
    data = json.load(f)

print("Before")
print(data["dimuon_mass"]['binning_linspace'])
# Now you can access it like a dictionary
data["dimuon_mass"]['binning_linspace'] = [30, 1000, 50]

with open("src/lib/histogram/plot_settings_test.json", "w") as f:
    json.dump(data, f, indent=4)

print("After")
print(data["dimuon_mass"]['binning_linspace'])