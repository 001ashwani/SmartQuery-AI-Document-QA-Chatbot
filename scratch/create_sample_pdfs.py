import os
import subprocess
import sys

# Install reportlab if not available
try:
    import reportlab
except ImportError:
    print("Installing reportlab for PDF generation...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    import reportlab

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_pdf(filename, pages_content):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    for page_num, text in enumerate(pages_content, start=1):
        c.setFont("Helvetica", 12)
        c.drawString(100, height - 100, f"Document: {os.path.basename(filename)}")
        c.drawString(100, height - 120, f"Page: {page_num}")
        
        # Write text content
        y = height - 200
        for line in text.split("\n"):
            c.drawString(100, y, line)
            y -= 20
            
        c.showPage()
    c.save()
    print(f"Created PDF: {filename}")


if __name__ == "__main__":
    doc_a_content = [
        "SmartQuery is a multiple PDF Q&A system.\nIt allows users to upload multiple documents at once.",
        "It uses a FAISS vector store to store and index text chunks for fast similarity search."
    ]
    doc_b_content = [
        "OpenAI embeddings are used to convert text chunks into vector representations.",
        "The chatbot retrieves the top k most similar chunks and passes them to GPT to generate answers."
    ]
    
    create_pdf("sample_docs/doc_a.pdf", doc_a_content)
    create_pdf("sample_docs/doc_b.pdf", doc_b_content)
