import yaml

year = "2022preEE"
# Load the YAML file
with open('../configs/datasets/dataset_nanoAODv12_zll.yaml', 'r') as file:
    data = yaml.safe_load(file)  # Use safe_load for security

# Print the parsed data
print(type(data))
print(data["years"][year])

sampleDict = data["years"][year]

# Data
print(f'''
## {year}
### Data
| Sample    | DAS |
| --------- | --- |''')
for sampleGroup, sampleDasList in sampleDict["Data"].items():
    print(f"| {sampleGroup[-1]} | {sampleDasList[0]}") # sampleGroup[-1] grabs only the letter label of dataset, for example in yaml, it is data_B, here it is just B.
    if len(sampleDasList) > 1:
        for das in sampleDasList[1:]:
            print(f"|   | {das}")

# Background
print(f'''
### Background MC
| Sample Group | Sample | DAS |
| ------------ | ------ | --- |''')
for sampleGroup, sampleDasDict in sampleDict.items():
    if sampleGroup == "Data":
        continue
    sampleList = list(sampleDasDict.keys())
    firstSample = sampleList[0]
    if type(sampleDasDict[firstSample]) == str:
        print(f"| {sampleGroup} | {firstSample} | {sampleDasDict[firstSample]}")
    else:
        print(f"| {sampleGroup} | {firstSample} | {sampleDasDict[firstSample][0]}")
        for das in sampleDasDict[firstSample][1:]:
            print(f"|   |    | {das}")
    if len(sampleList) > 1:
        for sample in sampleList[1:]:
            if type(sampleDasDict[sample]) == str:
                print(f"| {sampleGroup} | {sample} | {sampleDasDict[sample]}")
            else:
                print(f"| {sampleGroup} | {sample} | {sampleDasDict[sample][0]}")
                for das in sampleDasDict[sample][1:]:
                    print(f"|   |    | {das}")
    # if len(sampleDasDict[firstSample]) > 1:
    #     for das in sampleDasDict[firstSample][1:]:
    #         print(f"|   |    | {das}")
    # if len(sampleList) > 1:
    #     for sample in sampleList:
    #         print(f"|    | {sample} | {sampleDasDict[sample][0]}")
    #         if len(sampleDasDict[sample]) > 1:
    #             for das in sampleDasDict[sample][1:]:
    #                 print(f"|   |    | {das}")




