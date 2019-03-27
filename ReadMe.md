# Insight Coding Challenge
Submission for Insight Data Engineering Pharmacy Counting Coding Challenge

Jin Wang

# Table of Contents
1. [Source Files](README.md#Source-Files)
2. [Input](README.md#Input)
3. [Counting ](README.md#Counting)
4. [Sorting and Output](README.md#Sorting-and-Output)

# Source Files
prescribe_data.h   contains  data structure and class to address and store the input data as a list.
drug_data.h  contains data structure and class to store the output data as hashmap and sorted list of drug names.
couting_service.h   contains functions needed to implement the couting serices required by the project.
pharmacy_couting.cpp contains main function

# Input
The input file is a plain text file, it was scanned line by line using parse_infile function inside couting_service class. For each line (string), it's segmented into id_str, last_name, first_name,drug_name,drug_cost_str. We firstly check whether the id and drug_cost are valid numbers, if not, this line is skipped and coresponding error information is logged in to the logging file. After parsing this line, varaibles got are organized as an entry and put into a list.
![Reading Input File](pic/prescriber.png)

# Counting
For each drug, the number of Unique prescribes and total costs need to be counted. Thus, a hashmap, using the drug_name as key and other items as value is built. The hashmap allows us to locate, compare, and count the drug for each prescribe entry in the input data quickly. Since counting of unique prescriber is required, a hashset storing the prescibers for each drug is built. When couting the number of unique prescribers, whether the name of the prescriber can be found in the hashset is checked first, and number of prescribers increases by one if not found. At the same time, thenew prescriber is put into the hashset. In the meanwhile, total costs are updated every time.
The counting procedure is done by process_data_LinebyLine function inside counting_service.h
![Reading Input File](pic/drug.png)

# Sorting and Output
Since the output should be sorted in descending order according to total cost of the drug. We use a priority_queue to sort the pair<drug_name,drug_cost> in cost decreasing order. Then a ordered list of drug_name is obtained from the priority_queue. Based on the sorted list, and the hashmap with key of the drug name. We are able to output the processed data in to a text file.

