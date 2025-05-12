import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import pytesseract.pytesseract as pt

class data_loader:
    """Class to load and extract data from user-uploaded files."""
    def extract_text_from_pdf(self, pdf_files):
        text = ""
        if not pdf_files:
            st.error("No PDF files provided.")
            return text

        for pdf_file in pdf_files:
            # Validate file type
            if not pdf_file.name.lower().endswith('.pdf'):
                st.error(f"File {pdf_file.name} is not a valid PDF.")
                continue

            try:
                with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
                    for page_num, page in enumerate(doc, 1):
                        try:
                            extracted_text = page.get_text()
                            if not extracted_text.strip():
                                # Fallback to OCR for scanned/image-based PDFs
                                pix = page.get_pixmap()
                                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                extracted_text = pt.image_to_string(image)
                            text += extracted_text + "\n"
                        except Exception as e:
                            st.warning(f"Error processing page {page_num} in {pdf_file.name}: {str(e)}")
            except fitz.FileDataError:
                st.error(f"File {pdf_file.name} is corrupted or not a valid PDF.")
            except Exception as e:
                st.error(f"Error processing PDF {pdf_file.name}: {str(e)}")
        
        if not text.strip():
            st.warning("No text could be extracted from the provided PDFs.")
        return text

    def extract_text_from_image(self, image_file):
        """Extract text from an image file using Tesseract OCR."""
        if not image_file:
            st.error("No image file provided.")
            return ""

        # Validate file type
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
        if not image_file.name.lower().endswith(valid_extensions):
            st.error(f"File {image_file.name} is not a supported image format ({', '.join(valid_extensions)}).")
            return ""

        try:
            image = Image.open(image_file)
            text = pt.image_to_string(image)
            if not text.strip():
                st.warning(f"No text could be extracted from {image_file.name}.")
            return text
        except pt.TesseractError as e:
            st.error(f"Tesseract OCR error for {image_file.name}: {str(e)}")
            return ""
        except Exception as e:
            st.error(f"Error processing image {image_file.name}: {str(e)}")
            return ""