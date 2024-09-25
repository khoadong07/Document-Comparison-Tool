from flask import Blueprint, render_template, request, redirect
from .utils import read_txt, generate_side_by_side_html, process_pdf, ocr_norm_llm
import os

main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

@main.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')

        if not file1 or not file2 or file1.filename == '' or file2.filename == '':
            return redirect(request.url)

        # Save file1 for comparison
        content1 = read_txt(file1)

        # Handle the PDF OCR case
        if file2.filename.lower().endswith('.pdf'):
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            if not os.path.exists(OUTPUT_FOLDER):
                os.makedirs(OUTPUT_FOLDER)

            # Save the uploaded PDF file temporarily
            pdf_path = os.path.join(UPLOAD_FOLDER, file2.filename)
            file2.save(pdf_path)

            # Process the PDF: OCR and normalization
            ocr_output_path = os.path.join(OUTPUT_FOLDER, 'output_contract_text.txt')
            process_pdf(pdf_path, ocr_output_path)

            normalized_output_path = os.path.join(OUTPUT_FOLDER, 'normalized_output.txt')
            ocr_norm_llm(ocr_output_path, normalized_output_path)

            # Read the normalized text file and split into paragraphs
            with open(normalized_output_path, 'r', encoding='utf-8') as file:
                content = file.read()  # Read the file content

            # Split the content by lines and clean up
            paragraphs = content.splitlines()  # Split the text into lines
            content2 = [para.strip() for para in paragraphs if para.strip()]

            # Generate side-by-side HTML comparison
            side_by_side_html1, side_by_side_html2 = generate_side_by_side_html(content1, content2, file1.filename, "OCR Text")

            return render_template('index.html', side_by_side_html1=side_by_side_html1, side_by_side_html2=side_by_side_html2)

        else:
            # Handle file comparison as before if not a PDF
            content2 = read_txt(file2)
            side_by_side_html1, side_by_side_html2 = generate_side_by_side_html(content1, content2, file1.filename, file2.filename)

            return render_template('index.html', side_by_side_html1=side_by_side_html1, side_by_side_html2=side_by_side_html2)

    return render_template('index.html')
