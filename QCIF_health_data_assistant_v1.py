##### ------ IMPORT FUNCTIONS + SETUP CODE - START ------- ####

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import PyPDF2
from openai import OpenAI
import fitz  # PyMuPDF
import io
import os
from PIL import Image
import base64
import json
#import psycopg2  # (Optional)
import pandas as pd

##### ------ DEFINE FUNCTIONS - START ------- ####

def extract_text_from_pdf(pdf_path, start_page=None, end_page=None):
    """
    Extract text from a PDF file along with page numbers.
    Parameters:
    - pdf_path (str): The path to the PDF file.
    - start_page (int): The starting page number (inclusive).
    - end_page (int): The ending page number (inclusive).
    Returns:
    - list of tuples: Each tuple contains the page number and the extracted text.
    """
    pages = []
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(reader.pages)
        # Adjust page range to ensure valid bounds
        start_page = max(start_page, 1) if start_page else 1
        end_page = min(end_page, total_pages) if end_page else total_pages
        for page_num, page in enumerate(reader.pages, start=1):
            if start_page <= page_num <= end_page:
                pages.append((page_num, page.extract_text()))
    return pages

# Initialize the OpenAI client
client = OpenAI()

##### ------ DEFINE JSON SCHEMA MODELS - START ------- ####

# 1. Core Legal/Regulatory Features
class RegulatoryScope(BaseModel):
    comprehensive_vs_sector_specific: Optional[str] = None
    public_vs_private_sector: Optional[str] = None
    special_data_categories_included: Optional[List[str]] = None

class RegulatoryOversightAndEnforcement(BaseModel):
    regulatory_authorities: Optional[List[str]] = None
    oversight_mechanisms: Optional[str] = None
    penalties_for_non_compliance: Optional[str] = None

class CoreLegalRegulatoryFeatures(BaseModel):
    presence_of_data_protection_laws: bool
    key_laws_and_regulatory_instruments: List[str]
    sector_specific_laws: List[str]
    regulatory_scope: RegulatoryScope
    regulatory_oversight_and_enforcement: RegulatoryOversightAndEnforcement

# 2. Consent & Data Collection Models
class ConsentMechanisms(BaseModel):
    type_of_consent_required: Optional[str] = None
    secondary_use_permissions: Optional[str] = None

class DataCollectionAndRetention(BaseModel):
    limits_on_data_retention: Optional[str] = None
    data_minimization_requirements: Optional[str] = None

class ConsentAndDataCollectionModels(BaseModel):
    consent_mechanisms: ConsentMechanisms
    data_collection_and_retention: DataCollectionAndRetention

# 3. Data Protection & Privacy Safeguards
class SecurityMeasures(BaseModel):
    data_anonymization_and_de_identification: Optional[str] = None
    data_breach_notification_rules: Optional[Dict[str, Any]] = None

class CrossBorderDataTransfers(BaseModel):
    restrictions: Optional[str] = None
    mechanisms: Optional[str] = None

class DataProtectionAndPrivacySafeguards(BaseModel):
    security_measures: SecurityMeasures
    cross_border_data_transfers: CrossBorderDataTransfers

# 4. Research-Friendliness & Regulatory Flexibility
class NationalHealthDataInfrastructure(BaseModel):
    centralized_health_data_repositories: Optional[str] = None
    access_conditions: Optional[str] = None

class ResearchFriendlinessAndRegulatoryFlexibility(BaseModel):
    explicit_research_exemptions: Optional[str] = None
    ethics_review_and_oversight: Optional[str] = None
    regulatory_sandboxes: Optional[str] = None
    public_vs_private_research_impact: Optional[str] = None
    national_health_data_infrastructure: NationalHealthDataInfrastructure

# 5. Clinical Trials & Telehealth Regulations
class ClinicalTrialsDataProtection(BaseModel):
    legal_basis_for_processing: Optional[str] = None
    special_handling_of_pharmacovigilance_data: Optional[str] = None
    key_coded_data_regulations: Optional[str] = None
    international_transfer_restrictions: Optional[str] = None

class TelehealthDataProtection(BaseModel):
    regulatory_approach: Optional[str] = None
    data_security_standards: Optional[str] = None
    cross_border_telehealth_regulations: Optional[str] = None

class ClinicalTrialsAndTelehealthRegulations(BaseModel):
    clinical_trials_data_protection: ClinicalTrialsDataProtection
    telehealth_data_protection: TelehealthDataProtection

# 6. Institutional & Political Economy Considerations
class InstitutionalAndPoliticalEconomyConsiderations(BaseModel):
    regulatory_certainty_and_stability: Optional[str] = None
    public_trust_in_data_governance: Optional[str] = None
    transparency_and_governance_quality: Optional[str] = None
    lobbying_influences: Optional[str] = None

# Overall schema for the legal document
class LegalDocumentData(BaseModel):
    country: str
    core_legal_regulatory_features: CoreLegalRegulatoryFeatures
    consent_and_data_collection_models: ConsentAndDataCollectionModels
    data_protection_and_privacy_safeguards: DataProtectionAndPrivacySafeguards
    research_friendliness_and_regulatory_flexibility: ResearchFriendlinessAndRegulatoryFlexibility
    clinical_trials_and_telehealth_regulations: ClinicalTrialsAndTelehealthRegulations
    institutional_and_political_economy_considerations: InstitutionalAndPoliticalEconomyConsiderations


##### ------ DEFINE EXTRACTION FUNCTIONS - START ------- ####

#--- Function #1
def extract_legal_document_data_from_text(text_input, temperature_setting=0.7, max_tokens_setting=None, top_p_setting=1, presence_penalty_setting=0, n_setting=1, frequency_penalty_setting=0, logprobs_setting=False, model_setting="gpt-4o-mini", chain_of_thought=True, guiding_principles="Fact-driven and evidence-based, grounded in accurate data and reliable sources."):
    """
    Extracts structured legal and regulatory data from the supplied text based on a detailed checklist.
    Key Parameters:
    - text_input (str): The raw text data to be processed.
    - temperature_setting (float): Temperature setting for model creativity.
    - model_setting (str): The model to be used for the task.
    - chain_of_thought (bool): Whether to include chain-of-thought reasoning in the response.
    - guiding_principles (str): Customizable guiding principles for the output.
    Returns:
    - dict: Extracted structured data in JSON format.
    """
    system_prompt = f"""
    You are an AI assistant tasked with extracting structured legal and regulatory data from legal/statutory documents.
    Follow this checklist exactly:
    1. Core Legal/Regulatory Features:
       - Presence of Data Protection Laws: Identify if explicit laws/policies govern data protection, healthcare data, clinical trials, and telehealth.
       - Key Laws & Regulatory Instruments: (e.g., GDPR (EU), HIPAA (USA), PDPA (Singapore), Privacy Act 1988 (Australia)).
       - Sector-Specific Laws: Are there healthcare-specific data protection rules? (e.g., My Health Records Act in Australia).
       - Regulatory Scope:
           - Comprehensive vs. Sector-Specific: Does the law cover all industries or only healthcare?
           - Public vs. Private Sector: Does the regulation apply to public and private healthcare data?
           - Inclusion of Special Data Categories (e.g., genomic data, AI-based healthcare data, IoMT, wearable devices; clinical trial data; pharmacovigilance requirements).
       - Regulatory Oversight & Enforcement:
           - Regulatory Authorities (e.g., OAIC, UK ICO).
           - Oversight Mechanisms: Are regulators independent or government-controlled? Are there sector-specific oversight mechanisms?
           - Penalties for Non-Compliance: (fines, criminal liability, administrative penalties).
    2. Consent & Data Collection Models:
       - Consent Mechanisms: Type of consent (opt-in/opt-out, explicit consent for research, parental consent for minors), and secondary use permissions.
       - Data Collection & Retention: Limits on data retention and data minimization requirements.
    3. Data Protection & Privacy Safeguards:
       - Security Measures: Data anonymization/pseudonymization requirements, data breach notification rules (timeframe and sanctions).
       - Cross-Border Data Transfers: Restrictions and special agreements.
    4. Research-Friendliness & Regulatory Flexibility:
       - Explicit Research Exemptions, Ethics Review & Oversight, Regulatory Sandboxes, Public vs. Private Research Impact.
       - National Health Data Infrastructure: Centralized health data repositories and access conditions.
    5. Clinical Trials & Telehealth Regulations:
       - Clinical Trials Data Protection: Legal basis, handling of pharmacovigilance data, pseudonymization rules, international transfer restrictions.
       - Telehealth Data Protection: Recognition of telehealth as a service, data security standards, cross-border regulations.
    6. Institutional & Political Economy Considerations:
       - Regulatory Certainty & Stability, Public Trust in Data Governance, Transparency & Governance Quality, and Lobbying Influences.
    Use only the information explicitly stated in the supplied text.
    Output the result as a JSON object conforming to the LegalDocumentData schema.
    Guiding principles: {guiding_principles}
    """
    # Dynamic user prompt to provide specific instructions
    user_content = f"Extract structured legal and regulatory information based on the provided checklist from the following text: {text_input}"
    if chain_of_thought:
        user_content += " Let's think step-by-step."
    # Call to the language model for processing
    response = client.beta.chat.completions.parse(
        model=model_setting,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        response_format=LegalDocumentData,
        max_tokens=max_tokens_setting,
        temperature=temperature_setting,
        top_p=top_p_setting,
        presence_penalty=presence_penalty_setting,
        n=n_setting,
        frequency_penalty=frequency_penalty_setting,
        logprobs=logprobs_setting,
    )
    return response.choices[0].message.content


#--- Function #2
def save_legal_data_to_excel(legal_data: dict, output_file: str = "legal_data.xlsx"):
    """
    Saves the legal document data (structured as a dictionary) to an Excel file.
    Each top-level section of the legal data will be saved on a separate sheet.
    """
    with pd.ExcelWriter(output_file) as writer:
        sheet_created = False
        # Iterate over each top-level key in the JSON dictionary
        for section_name, section_content in legal_data.items():
            try:
                # Attempt to flatten the section using json_normalize if it's a dict or list
                if isinstance(section_content, dict):
                    # Convert dict to a single-row DataFrame
                    df = pd.json_normalize(section_content, sep="_")
                elif isinstance(section_content, list):
                    df = pd.DataFrame(section_content)
                else:
                    # Fallback: wrap non-dict, non-list content in a list
                    df = pd.DataFrame([{section_name: section_content}])
            except Exception as e:
                print(f"Error normalizing section '{section_name}': {e}")
                # As fallback, create a one-row dataframe with the raw data
                df = pd.DataFrame({section_name: [str(section_content)]})
            # If the DataFrame is empty, add a message row
            if df.empty:
                df = pd.DataFrame({"message": [f"No data available for {section_name}"]})
            # Ensure the sheet name is within Excel's 31-character limit
            sheet_name = section_name[:31]
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            sheet_created = True
        # If no sheet was created, add a default sheet.
        if not sheet_created:
            pd.DataFrame({"message": ["No legal data available"]}).to_excel(writer, index=False, sheet_name="Sheet1")
    print(f"Legal data has been saved to {output_file}")


#--- Function #3
def save_legal_data_to_wide_excel(legal_data: dict, output_file: str = "legal_data_wide.xlsx"):
    """
    Flattens the entire legal data JSON into a single-row DataFrame and saves it as a wide-format Excel file.
    """
    # Flatten the JSON object using pd.json_normalize with a separator for nested keys
    df = pd.json_normalize(legal_data, sep='_')
    # Write the DataFrame to a single Excel sheet
    df.to_excel(output_file, index=False)
    print(f"Legal data has been saved to {output_file}")


##### ------ MAIN CODE - START ------- ####

# Extract the text from the pdfs
pages = extract_text_from_pdf("data_protection_laws_and_regulations_australia.pdf", start_page=2,
                              end_page=47)

# Apply the function to the text extracted from the pdfs
legal_data_json = extract_legal_document_data_from_text(str(pages), temperature_setting=0.7, max_tokens_setting=None,
                                               top_p_setting=1, presence_penalty_setting=0, n_setting=1,
                                               frequency_penalty_setting=0, logprobs_setting=False,
                                               model_setting="gpt-4o", chain_of_thought=True,
                                               guiding_principles="Fact-driven and evidence-based, grounded in accurate data and reliable sources.")

# Load the JSON string into a Python dictionary
legal_data = json.loads(legal_data_json)

# Save the data to an Excel file
#save_legal_data_to_excel(legal_data, output_file="legal_document_data.xlsx")
save_legal_data_to_wide_excel(legal_data, output_file="legal_document_data_wide.xlsx")

##### ------ MAIN CODE - END ------- ####