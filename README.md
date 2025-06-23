SmartABCPeptide

A machine learning-powered web application for the classification of anticancer peptides (ACPs) and non-ACPs from protein/peptide sequences.

Developed by System Biology LaboratoryIndian Institute of Information Technology Allahabad, Prayagraj, Uttar Pradesh, India.

🎯 Features

Single or batch prediction (supports multi-FASTA or one-sequence-per-line input)

Input methods: manual paste or FASTA file upload

Downloadable CSV results

Interactive modern web UI using Streamlit

🚀 Quick Start

1. Clone the Repository

git clone https://github.com/yourusername/abcp-classifier.git
cd SmartABCPeptide/SmartABCPeptide/

2. Install Conda (Miniconda or Anaconda)

Download and install Miniconda:
https://www.anaconda.com/docs/getting-started/miniconda/main

3. Create and Activate Environment

conda env create -f environment.yaml
conda activate abcp_env

4. Run the App

streamlit run app.py

Open http://localhost:8501 in your browser and start classifying.


🧪 Example Usage

Paste multi-FASTA or raw text sequences:

>Peptide1
ALLK
>Peptide2
ALLKK

Or one sequence per line:

ALLK
ALLKK

Click Classify to view and download predictions in CSV format.


📬 Contact

For support, contact: Prabhat Tripathi (pmb2022001@iiita.ac.in) System Biology LaboratoryIndian Institute of Information Technology Allahabad

Enjoy using ABCP Classifier!
