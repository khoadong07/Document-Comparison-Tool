import os
import re
from io import BytesIO
from docx import Document
from dotenv import load_dotenv
from unidecode import unidecode
import os
import cv2
import pytesseract
from pdf2image import convert_from_path
import numpy as np
from openai import OpenAI
import difflib


load_dotenv()

api_key = os.getenv('DEEP_INFRA_API')
base_url = os.getenv('DEEP_INFRA_URL')

openai = OpenAI(api_key=api_key, base_url=base_url)

def normalize_text(text):
    text = unidecode(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def read_txt(file):
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext == '.docx':
        content = read_docx(file)
    else:
        content = file.read().decode('utf-8')
        paragraphs = content.splitlines()
        content = [para.strip() for para in paragraphs if para.strip()]

    return content


def read_docx(file):
    doc = Document(BytesIO(file.read()))
    content = []

    for para in doc.paragraphs:
        if para.text.strip():
            numbering = ''

            if para.style.name.startswith('List'):
                numbering = para.style.name

            content.append(f"{numbering} {para.text.strip()}")
    print(content)
    return content

# def generate_side_by_side_html(content1, content2, filename1, filename2):
#     original_content1 = content1.copy()
#     original_content2 = content2.copy()
#
#     normalized1 = [normalize_text(line) for line in content1]
#     normalized2 = [normalize_text(line) for line in content2]
#
#     diff = difflib.ndiff(normalized1, normalized2)
#
#     html_content1 = f"<div><h3>{filename1}</h3>"
#     html_content2 = f"<div><h3>{filename2}</h3>"
#
#     for line, orig1, orig2 in zip(diff, original_content1, original_content2):
#         if line.startswith("- "):
#             html_content1 += f"<span class='diff-removed'>{orig1}</span><br>"
#             html_content2 += f"<span class='diff-removed'>{orig2}</span><br>"
#         elif line.startswith("+ "):
#             html_content1 += f"{orig1}<br>"
#             html_content2 += f"<span class='diff-added'>{orig2}</span><br>"
#         else:
#             html_content1 += f"{orig1}<br>"
#             html_content2 += f"{orig2}<br>"
#
#     html_content1 += "</div>"
#     html_content2 += "</div>"
#
#     return html_content1, html_content2

def generate_side_by_side_html(content1, content2, filename1, filename2):
    original_content1 = content1.copy()
    original_content2 = content2.copy()

    normalized1 = [normalize_text(line) for line in content1]
    normalized2 = [normalize_text(line) for line in content2]

    diff = difflib.ndiff(normalized1, normalized2)

    html_content1 = f"<div><h3>{filename1}</h3>"
    html_content2 = f"<div><h3>{filename2}</h3>"

    index1 = 0
    index2 = 0

    for line in diff:
        if line.startswith("- "):
            if index1 < len(original_content1):
                html_content1 += f"<span class='diff-removed'>{original_content1[index1]}</span><br>"
            else:
                html_content1 += "<span class='diff-removed'></span><br>"
            html_content2 += "<span class='diff-removed'></span><br>"
            index1 += 1
        elif line.startswith("+ "):
            html_content1 += "<span class='diff-added'></span><br>"
            if index2 < len(original_content2):
                html_content2 += f"<span class='diff-added'>{original_content2[index2]}</span><br>"
            else:
                html_content2 += "<span class='diff-added'></span><br>"
            index2 += 1
        else:
            if index1 < len(original_content1):
                html_content1 += f"{original_content1[index1]}<br>"
            else:
                html_content1 += "<br>"

            if index2 < len(original_content2):
                html_content2 += f"{original_content2[index2]}<br>"
            else:
                html_content2 += "<br>"

            index1 += 1
            index2 += 1

    html_content1 += "</div>"
    html_content2 += "</div>"

    return html_content1, html_content2


# def generate_side_by_side_html(content1, content2, filename1, filename2):
#     # Create copies of the original content (with accents) to display in the output
#     original_content1 = content1.copy()
#     original_content2 = content2.copy()
#
#     # Normalize both sets of content (without accents, lowercase) for comparison
#     normalized1 = [normalize_text(line) for line in content1]
#     normalized2 = [normalize_text(line) for line in content2]
#
#     # Use difflib to get the differences between the two sets of normalized text
#     diff = difflib.ndiff(normalized1, normalized2)
#
#     # Prepare the HTML containers for the two side-by-side comparisons
#     html_content1 = f"<div><h3>Original Document</h3>"
#     html_content2 = f"<div><h3>Changed Docuemnt</h3>"
#
#     # Loop through the differences and highlight the changes
#     for line in diff:
#         # We use the line number for reference (not the normalized line)
#         orig1, orig2 = None, None
#
#         # Fetch the original lines that correspond to the difference
#         if line.startswith("- "):  # Line is removed from file 1
#             orig1 = original_content1.pop(0) if original_content1 else ""
#             html_content1 += f"<span class='diff-removed'>{orig1}</span><br>"
#             html_content2 += "<br>"  # No corresponding line in file 2
#         elif line.startswith("+ "):  # Line is added in file 2
#             orig2 = original_content2.pop(0) if original_content2 else ""
#             html_content1 += "<br>"  # No corresponding line in file 1
#             html_content2 += f"<span class='diff-added'>{orig2}</span><br>"
#         else:  # Line is the same in both files
#             orig1 = original_content1.pop(0) if original_content1 else ""
#             orig2 = original_content2.pop(0) if original_content2 else ""
#             html_content1 += f"{orig1}<br>"
#             html_content2 += f"{orig2}<br>"
#
#     # Close the div containers for the HTML content
#     html_content1 += "</div>"
#     html_content2 += "</div>"
#
#     return html_content1, html_content2
def normalize_text_compare(text):
    """
    Normalize the text by:
    1. Removing special characters.
    2. Converting to lowercase.
    3. Removing extra spaces.
    4. Converting Vietnamese text to non-diacritic (without accents).
    """
    text = unidecode(text)  # Convert Vietnamese characters to non-diacritic
    text = re.sub(r'[^\w\s]', '', text)  # Remove all special characters except for letters and numbers
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip().lower()  # Return lowercased stripped text


# def search_in_content(content1, content2):
#     """
#     Perform a search for each item in content1 in content2.
#     This helps find matching content in both directions.
#     """
#     normalized_content2 = [normalize_text(line) for line in content2]
#     matched_in_content2 = []
#
#     for line1 in content1:
#         normalized_line1 = normalize_text(line1)
#         found = any(normalized_line1 == line2 for line2 in normalized_content2)
#         matched_in_content2.append(found)
#
#     return matched_in_content2
#
#
# def generate_side_by_side_html(content1, content2, filename1, filename2):
#     # Normalize the text for comparison
#     normalized1 = [normalize_text_compare(line) for line in content1]
#     normalized2 = [normalize_text_compare(line) for line in content2]
#
#     # Create lists to store the matches
#     matched_in_content2 = search_in_content(content1, content2)
#     matched_in_content1 = search_in_content(content2, content1)
#
#     # Prepare the HTML containers for the two side-by-side comparisons
#     html_content1 = f"<div><h3>{filename1}</h3>"
#     html_content2 = f"<div><h3>{filename2}</h3>"
#
#     # Iterate through content1 and check if there's a match in content2
#     for idx, (line1, matched) in enumerate(zip(content1, matched_in_content2)):
#         if matched:
#             # Check if there's a corresponding line in content2, and it’s not a modification
#             corresponding_line2 = content2[normalized2.index(normalize_text_compare(line1))]
#             if normalize_text_compare(line1) == normalize_text_compare(corresponding_line2):
#                 html_content1 += f"{line1}<br>"
#                 html_content2 += f"{corresponding_line2}<br>"
#             else:
#                 # Treat it as a modification
#                 html_content1 += f"<span class='diff-removed'>{line1}</span><br>"
#                 html_content2 += f"<span class='diff-added'>{corresponding_line2}</span><br>"
#         else:
#             # Line removed from content1
#             html_content1 += f"<span class='diff-removed'>{line1}</span><br>"
#             html_content2 += "<br>"  # Nothing matched in content2
#
#     # Now check if content2 has any extra lines not in content1 (added lines)
#     for idx, (line2, matched) in enumerate(zip(content2, matched_in_content1)):
#         if not matched:
#             html_content1 += "<br>"  # No corresponding line in content1
#             html_content2 += f"<span class='diff-added'>{line2}</span><br>"
#
#     # Close the div containers for the HTML content
#     html_content1 += "</div>"
#     html_content2 += "</div>"
#
#     return html_content1, html_content2

# OCR functions
def remove_tables(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    mask = np.ones(gray.shape, dtype=np.uint8) * 255

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(mask, (x1, y1), (x2, y2), 0, 10)

    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def ocr_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_image, lang='vie')
    return text

def process_pdf(pdf_path, output_text_path):
    images = convert_from_path(pdf_path)

    with open(output_text_path, 'w', encoding='utf-8') as f:
        for page_num, image in enumerate(images):
            print(f"page_num: {page_num}")
            open_cv_image = np.array(image)[:, :, ::-1].copy()
            processed_image = remove_tables(open_cv_image)
            page_text = ocr_image(processed_image)
            f.write(page_text)
            f.write("\n")

def ocr_norm_llm(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            ocr_text = file.read()
    except FileNotFoundError:
        print(f"File not found: {input_file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    prompt = f"""
        You are an AI language model. Your task is to clean up and correct OCR text errors in the following document.

        Please ensure that:
        - All capitalization is correct.
        - All spelling errors and garbled characters are fixed.
        - Proper punctuation is restored.
        - The document structure and formatting are maintained (including headings, numbered sections, etc.).
        - Ensure names, dates, and legal terminology are clear and accurate.

        Here is the document that needs correction:

        {ocr_text}

        Please return the corrected and cleaned text with no additional explanations.
    """

    chat_completion = openai.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    normalized_text = chat_completion.choices[0].message.content

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(normalized_text)

    print(chat_completion.choices[0].message.content)
    print(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)



