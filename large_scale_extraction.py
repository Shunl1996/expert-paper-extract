import langfun as lf # we use langfun to handle LLMs and PDF inputs
import os
import json
import re
import pandas as pd
import time
from dotenv import load_dotenv
load_dotenv()  # reads .env in project root directory

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

prompt = """
Read the PDF file and extract the following parameters for all high-entropy alloys discussed in the results section:

1. name (string, e.g., "AlCoCrFeNi")
2. nominal_composition (string, representing the stoichiometric ratio of each element, e.g., "Al1.0Co1.0Cr1.0Fe1.0Ni1.0". If an element's ratio is not explicitly stated, assume 1.0)
3. measured_composition (string, exactly as written in the paper)
4. lattice_constant (float, in angstroms, rounded to 3 decimal places)
5. phases (string, e.g., "BCC")
6. alloy_condition (string, e.g., "As-Cast")
7. doi (string)

Extract parameters primarily from the text. Use data from tables only if the text data is incomplete. Use figures as a last resort. If a parameter is truly missing from the PDF for a given alloy, explicitly report it as "Not found" rather than omitting it.

For each parameter, include a confidence score (0-100) indicating your certainty in the extracted information. Consider a score of 90 or above as high confidence.

The output should be a list of JSON objects, one for each alloy discussed in the paper, in the following format:

[
    {
        "name": "AlloyName",
        "nominal_composition": "Element11.0Element21.0...",
        "measured_composition": "Composition as written",
        "lattice_constant": X.XXX,
        "phases": "Phase1,Phase2,...",
        "alloy_condition": "condition",
        "doi": "DOI",
        "confidence_scores": {
            "name": XX,
            "nominal_composition": XX,
            "measured_composition": XX,
            "lattice_constant": XX,
            "phases": XX,
            "alloy_condition": XX,
            "doi": XX
        }
    },
    ...
]

Include an alloy in the output only if it is explicitly discussed in the results section. Ensure that the output format and data closely match the provided schema. If information for a specific parameter is not available, use "Not found" and assign a low confidence score.

Example of correct output:
[
    {
        "name": "HfNbTaTiZr",
        "nominal_composition": "Hf1.0Nb1.0Ta1.0Ti1.0Zr1.0",
        "measured_composition": "Hf20.8Nb18.9Ta20.2Ti20.2Zr19.9",
        "lattice_constant": 3.414,
        "phases": "BCC",
        "alloy_condition": "As-Cast",
        "doi": "10.1016/j.jallcom.2014.11.064",
        "confidence_scores": {
            "name": 100,
            "nominal_composition": 90,
            "measured_composition": 95,
            "lattice_constant": 100,
            "phases": 100,
            "alloy_condition": 95,
            "doi": 100
        }
    }
]

Example of output with missing data:
[
    {
        "name": "AlCoCrFeNi",
        "nominal_composition": "Al1.0Co1.0Cr1.0Fe1.0Ni1.0",
        "measured_composition": "Not found",
        "lattice_constant": 3.567,
        "phases": "FCC",
        "alloy_condition": "Not found",
        "doi": "10.1016/j.example.2023.01.001",
        "confidence_scores": {
            "name": 100,
            "nominal_composition": 90,
            "measured_composition": 0,
            "lattice_constant": 95,
            "phases": 100,
            "alloy_condition": 0,
            "doi": 100
        }
    }
]

    """

def response_to_json(response,filename):
    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    match2 = re.search(r"(\[.*?\])", response,re.DOTALL)
    match3 = re.search(r"(\{.*?\})", response,re.DOTALL)
    if match:
        json_content = match.group(1).strip()
        try:
            parsed_data = json.loads(json_content)
            if isinstance(parsed_data, list):  # Ensure it's a list of dictionaries
                with open(filename + ".json", "w") as f:
                    json.dump(parsed_data, f, indent=4)
            # df = pd.DataFrame(parsed_data)
        except json.JSONDecodeError as e:
            print("Invalid JSON:", e)
    elif match2:
        json_content = match2.group(1).strip()
        try:
            parsed_data = json.loads(json_content)
            if isinstance(parsed_data, list):  # Ensure it's a list of dictionaries
                with open(filename + ".json", "w") as f:
                    json.dump(parsed_data, f, indent=4)
            # df = pd.DataFrame(parsed_data)
        except json.JSONDecodeError as e:
            print("Invalid JSON:", e)
    elif match3:
        json_content = match3.group(1).strip()
        try:
            parsed_data = json.loads(json_content)
            with open(filename + ".json", "w") as f:
                json.dump(parsed_data, f, indent=4)
            # df = pd.DataFrame(parsed_data)
        except json.JSONDecodeError as e:
            print("Invalid JSON:", e)
    else:
        print("⚠️ Text does not have the expected format.")
        with open(filename + "_raw.txt", "w") as f:
            f.write(response)
        print(f"📝 Entire response saved to {filename}_raw.txt")


if __name__ == "__main__":
    # i = 0
    # loop over the pdfs in the papers directory
    pdf_path = "YOUR_PDF_DIRECTORY_PATH_HERE/"  # replace with your pdf directory path
    for file in os.listdir(pdf_path):
        if file.endswith(".pdf"):
            filename = os.path.basename(file).split(".")[0]
            file_path = pdf_path + file
            print(file_path)
            if os.path.exists(filename + '.json') or os.path.exists(filename + "_raw.txt"):
                print(f"⚠️ File {filename} already exists. Skipping.")
                continue
            pdf_to_read = lf.PDF(file_path)
            # print(pdf_to_read.is_text)
            # pdf_to_read = pdf_to_read.to_text()
            #Here we use Claude 3.5 Sonnet which supports PDF input, and we use langfun to handle the PDF input and output
            try:
                r = lf.query(prompt= prompt + '{{PDF}}', PDF=pdf_to_read, lm=lf.llms.Claude35Sonnet_20241022())
            except:
                print("Unable to process this file")
                r = 'Unbale to process' + filename
            response_to_json(response=r,filename=filename)
            time.sleep(20)
