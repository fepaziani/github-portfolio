# github-portfolio
#This project trains a custom NER model using spaCy in Portuguese to extract specific entities (REF1 and REF2) from structured text data. The extracted entities are stored back in an Excel file for further data analysis.
#In the case this code was used to extract data from invoices


import spacy
import pandas as pd
from spacy.training import Example
import random

# Creates an empty spaCy model
nlp = spacy.blank("pt")

# Adds the NER component to the pipeline
ner = nlp.add_pipe("ner", name="custom_ner")

# Adds entity labels
ner.add_label("REF1")
ner.add_label("REF2")

# Training data for the custom NER model
TRAIN_DATA = [("2003           1    Payment referring to electricity bill for Subsede MONTE         0027           301049                            470.13                  470.13                          CARMELO - OUT/2022", {
        'entities': [(20, 82, 'REF1'), (196, 214, 'REF2')]
    }),
    ("2003               Payment for the 10,000 KM revision of the truck          0027           301030                          800.52                  800.52                          HILUX plate RCM6G24", {
        'entities': [(19, 77, 'REF1'), (190, 209, 'REF2')]
    }),
     ("Cost Center    Quantity 2003            1    Purchase of popsicles for distribution at the event  0027 301045 Maintenance of LT 800KV Xingu-Estreito", { 
        'entities': [(45, 103, 'REF1'), (116, 153, 'REF2')]
    }),
     ("R$79.45                            R$79.452004    1 Electricity bill for Eletrodo SE-Estreito (DECEMBER/2022)  27          3.01.049                                                                            --", { 
        'entities': [(52, 98, 'REF1'), (99, 114, 'REF2')]
    }),
     ("Hiring services for Urban and Synanthropic Pest Control 2003                1       (Disinsection) for the central office building area                        0027                301019                                       650.00                            650.00 Maintenance of LT 800KV Xingu -Estreito invoice: 1274 BMTE-TMA-O&M-DT-                -2022",{        
        'entities': [(0, 78, 'REF1'), (107, 180, 'REF2')]
    }),
     ("Hiring a company for specialized repair services of 2003          1     equipment and instruments.                                         0027            301031                         1,286.59                1,286.59 ",{        
        'entities': [(0, 64, 'REF1'), (85, 113, 'REF2')]
    }),
    ("2004           1     PAYMENT FOR MAINTENANCE ON DETECTOR                        27            101.031                          945.40                  945.4 SERIES AKDG00715 - SE-ESTREITO.      ",{        
        'entities': [(21, 65, 'REF1'), (166, 196, 'REF2')]
    }),
    (" 2003                        1         Payment for the 10,000 KM revision of the HILUX truck                                                                        0027                          301030                                                         692.90                                          692.90 plate QTQ7385",{        
        'entities': [(39, 103, 'REF1'), (323, 336, 'REF2')]
    }),
    ("  2003  ,  1 .  Hiring services for recreation at the event for the Maintenance of LT 800KV Xingu-Estreito  ",{        
        'entities': [(16, 77, 'REF1'), (79, 136, 'REF2')]
    }),
    ("2004          1     Payment for various materials for maintenance             27            3.01.028                          209.10                  209.10 for the Estreito Substation.",{        
        'entities': [(20, 78, 'REF1'), (170, 193, 'REF2')]
    }),
]

# Starts training the model
optimizer = nlp.begin_training()

for itn in range(1000):
    random.shuffle(TRAIN_DATA)
    losses = {}
    for text, annotations in TRAIN_DATA:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], drop=0.5, losses=losses)
    print(f"Iteration {itn}, Loss: {losses}")

# Saves the trained model
nlp.to_disk("model")

# Loads the trained model for testing
nlp = spacy.load("model")
doc = nlp("2004           1    Payment for cleaning material for maintenance and conservation of the Estreito Substation building.")
for ent in doc.ents:
    print(ent.text, ent.label_)

# Post-processing: combines extracted entities
extracted_text = ""
for ent in doc.ents:
    if ent.label_ in ['REF1', 'REF2']:
        extracted_text += ent.text + " "
extracted_text = extracted_text.strip()
print(extracted_text)

# Loads the trained model
nlp = spacy.load("model")

# Function to extract entities from text using the spaCy model
def extract_entities(text):
    doc = nlp(text)
    extracted_text = ""
    for ent in doc.ents:
        if ent.label_ in ['REF1', 'REF2']:
            extracted_text += ent.text + " "
    return extracted_text.strip()

# Path to the Excel file
excel_file = r"C:\Users\FPaziani\OneDrive - Alvarez and Marsal\testeM.xlsx"

# Loads the Excel file
df = pd.read_excel(excel_file)

# Processes each row and stores the result in column B
df['B'] = df['A'].apply(extract_entities)

# Saves the Excel file with the processed results
df.to_excel(excel_file, index=False)

print(f"File {excel_file} processed successfully.")
