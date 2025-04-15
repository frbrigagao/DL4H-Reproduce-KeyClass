##################################################################################
# Notes: This is code for reading in the raw MIMIC data and processing it into
# something that can be worked with in the pipeline. 
# TODO:
  # 1) Process text by removing stop words and creating a vocab
##################################################################################

library(stringr)
library(data.table)
library(dplyr)
library(lubridate)
library(caret)
Sys.setenv(TZ='UTC')

##################################################################################
# Load in Data
##################################################################################
notes = as_tibble(fread('/mimic_csv_files/NOTEEVENTS.csv'))# actual file use when testing is done

# Please check out the below rows in the clinical notes. The addendum one has information
# like DISCHARGE DIAGNOSES which gives away pretty easily if kept in the data.
# notes = filter(notes, ROW_ID %in% c(8772, 55972))
diagnoses = as_tibble(fread('/mimic_csv_files/DIAGNOSES_ICD.csv'))

##################################################################################
# Ensure consistent types for join operations
##################################################################################
# Convert HADM_ID to character in both datasets to ensure proper join
# On NOTEEVENTS.csv -> ROW_ID, SUBJECT_ID and HADM_ID are numbers
# On DIAGNOSES_ICD.csv -> ROW_ID, SUBJECT_ID are numbers. However, HADM_ID is a string.
notes = notes %>% mutate(HADM_ID = as.character(HADM_ID))
diagnoses = diagnoses %>% mutate(HADM_ID = as.character(HADM_ID), 
                                SUBJECT_ID = as.character(SUBJECT_ID))

##################################################################################
# Collapse all ICD9 codes for each admission. Order might not be preserved
##################################################################################
# OLD CODE:
# diagnoses = select(diagnoses, HADM_ID, SUBJECT_ID, ICD9_CODE)
# diagnoses = group_by_(diagnoses, .dots=c("HADM_ID","SUBJECT_ID"))
# diagnoses = summarise_each(diagnoses, funs(paste(., collapse = "-")))
# diagnoses = ungroup(diagnoses)

diagnoses = diagnoses %>%
  select(HADM_ID, SUBJECT_ID, ICD9_CODE) %>%
  group_by(HADM_ID, SUBJECT_ID) %>%
  summarize(ICD9_CODE = paste(ICD9_CODE, collapse = "-"), .groups = "drop")

##################################################################################
# Filter Notes
##################################################################################
# there are duplicated HADM_IDs so there are multiple discharge notes
# there must be a smart way to remove the duplicates, but for now
# might just take the one with the data which comes first
# OLD CODE:
# notes = filter(notes, CATEGORY == 'Discharge summary')
# notes = select(notes, HADM_ID, CHARTDATE, DESCRIPTION, TEXT)
# notes = mutate(notes, CHARTDATE = as.POSIXct(CHARTDATE))
# notes = group_by_(notes, .dots=c("HADM_ID"))
# notes = filter(notes, CHARTDATE == min(CHARTDATE))# take the first one because idk why. Have to understand why there are more than one discharge summaries
# notes = filter(notes, !duplicated(HADM_ID))# Still have duplicates on the same day. For right now removing can do better later.

notes = notes %>% 
  filter(CATEGORY == 'Discharge summary') %>%
  select(HADM_ID, CHARTDATE, DESCRIPTION, TEXT) %>%
  mutate(CHARTDATE = as.POSIXct(CHARTDATE)) %>%
  group_by(HADM_ID) %>%
  filter(CHARTDATE == min(CHARTDATE)) %>% 
  filter(!duplicated(HADM_ID)) %>%  
  ungroup()

##################################################################################
# Create note and ICD9 code matrix
##################################################################################
# OLD CODE
# notes = ungroup(notes); diagnoses = ungroup(diagnoses)

icd9NotesDataTable = right_join(x = diagnoses, y = notes, by = 'HADM_ID')


##################################################################################
# Extra filtering
##################################################################################
icd9NotesDataTable = filter(icd9NotesDataTable, ICD9_CODE != '')

# filter out any words not needed
# filter out and icd9 codes not needed

#####################################################################################
# Add two additional columns
# 1. Level 2 Categories: Process all ICD-9 codes and convert them to the 3 digit disease categories
# 2. ICD-9 Top Level Categories: process all ICD-9 codes and convert them to their parent categories (19)
# https://en.wikipedia.org/wiki/List_of_ICD-9_codes
#####################################################################################

icd9NotesDataTable$Level2ICD = NA
icd9NotesDataTable$TopLevelICD = NA
for (i in 1:nrow(icd9NotesDataTable)){
  #print(i)
  icd9row = str_split(icd9NotesDataTable$ICD9_CODE[i], '-')[[1]]
  #icd9List = (sapply(icd9row, function(x) substr(x,1, 3),USE.NAMES = F))
  #icd9NotesDataTable[i,9] = paste(icd9List, collapse = '-')
  icd9ListLevel2 = c()
  icd9ListTop = c()
  for (icd9 in icd9row){
    icd9row = icd9row[icd9row!= ""]
    if (length(icd9row) == 0) {
      next
    }
    else
    {
      if (substr(icd9, 1, 1) == "E"){
        icd9Level2 = substr(icd9, 1, 4)
        icd9Top = "cat:19"
      }
      else if (substr(icd9, 1,1) == "V"){
        icd9Level2 = substr(icd9, 1, 3)
        icd9Top = "cat:19"
      }
      else {
        icd9 = as.numeric(substr(icd9, 1, 3))
        if (icd9 >= 001 && icd9 <= 139) {
          icd9Top = "cat:1"
        }
        else if (icd9 >= 140 && icd9 <= 239){
          icd9Top = "cat:2"
        }
        else if(icd9 >= 240 && icd9 <= 279){
          icd9Top = "cat:3"
        }
        else if(icd9 >= 280 && icd9 <= 289){
          icd9Top = "cat:4"
        }
        else if (icd9 >= 290 && icd9 <= 319){
          icd9Top = "cat:5"
        }
        else if(icd9 >= 320 && icd9 <= 359){
          icd9Top = "cat:6"
        }
        else if(icd9 >= 360 && icd9 <= 389){
          icd9Top = "cat:7"
        }
        else if (icd9 >= 390 && icd9 <= 459){
          icd9Top = "cat:8"
        }
        else if (icd9 >= 460 && icd9 <= 519){
          icd9Top = "cat:9"
        }
        else if (icd9 >= 520 && icd9 <= 579){
          icd9Top = "cat:10"
        }
        else if (icd9 >= 580 && icd9 <= 629){
          icd9Top = "cat:11"
        }
        else if (icd9 >= 630 && icd9 <= 679){
          icd9Top = "cat:12"
        }
        else if (icd9 >= 680 && icd9 <= 709){
          icd9Top = "cat:13"
        }
        else if (icd9 >= 710 && icd9 <= 739){
          icd9Top = "cat:14"
        }
        else if (icd9 >= 740 && icd9 <= 759){
          icd9Top = "cat:15"
        }
        else if (icd9 >= 760 && icd9 <= 779){
          icd9Top = "cat:16"
        }
        else if (icd9 >= 780 && icd9 <= 799){   # ATTENTION: CODE DOESN´T MATCH Table 1 of the FasTag Paper
          icd9Top = "cat:17"                    # In the FasTag paper category 17 goes from 800-899
        }                                       #
        else if (icd9 >= 800 && icd9 <= 999){   #
          icd9Top = "cat:18"                    #
        }                                       #
      }
    }
    icd9ListTop = c(icd9ListTop, icd9Top)
    icd9ListLevel2 = c(icd9ListLevel2, icd9)
    }

  # icd9NotesDataTable[i,7] = paste(unique(icd9ListLevel2), collapse = '-')
  # icd9NotesDataTable[i,8] = paste(unique(icd9ListTop), collapse = '-')

  # Use direct assignment to columns to avoid indexing issues
  icd9NotesDataTable$Level2ICD[i] = paste(unique(icd9ListLevel2), collapse = '-')
  icd9NotesDataTable$TopLevelICD[i] = paste(unique(icd9ListTop), collapse = '-')

  #break;
}


##################################################################################
# Split into Training and Validation
# consider smarter splits which consider dates and admission IDs so you don't have 
# any admissions in train and test
##################################################################################
set.seed(42)  # Added for reproducibility
trainingFrac = 0.70 # According to paper 70/30 split for training/testing data.

trainingIdxs = sample.int(n = floor(nrow(icd9NotesDataTable)*trainingFrac), replace = FALSE)
icd9NotesDataTable_train = icd9NotesDataTable[trainingIdxs,]
icd9NotesDataTable_valid = icd9NotesDataTable[-trainingIdxs,]

##################################################################################
# Write to file
##################################################################################
# Output file structure:
# Columns: "","HADM_ID","SUBJECT_ID","ICD9_CODE","CHARTDATE","DESCRIPTION","TEXT","Level2ICD","TopLevelICD"
#                 0         1             2           3           4           5       6           7
# Sample Data:
# "1","100001","58526","25013-3371-5849-5780-V5867-25063-5363-4580-25043-40390-5853-25053-36201-25083-7078-V1351",2117-09-17,"Report","Text of the report","250-337-584-578-V5867-536-458-403-585-362-707-V1351","cat:3-cat:6-cat:11-cat:10-cat:19-cat:8-cat:7-cat:13"

write.csv(icd9NotesDataTable, '/intermediate_files/icd9NotesDataTable.csv')
write.csv(icd9NotesDataTable_train, 'intermediate_files/icd9NotesDataTable_train.csv')
write.csv(icd9NotesDataTable_valid, 'intermediate_files/icd9NotesDataTable_test.csv')

##################################################################################
# Write files in a way that is easily read by python
##################################################################################
# # open file here
# for(lineIdx in 1:nrow(icd9NotesDataTable_train)){
#   line = icd9NotesDataTable_train[lineIdx,]
#   # for each line 
#     # write one word
# }