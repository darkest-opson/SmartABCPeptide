# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from abcp_classification_function import classify_peptide

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="SmartABCPeptide",
    layout="centered",
    page_icon="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WjEbOoAAAAASUVORK5CYII="
)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# -------------------------------
# Sidebar navigation
# -------------------------------
st.sidebar.title("🧬 SmartABCPeptide")
page = st.sidebar.radio(
    "Navigation",
    ["ABCP Classification", "Peptide Design"]
)

# ======================================================
# Helper functions
# ======================================================
def parse_fasta(raw_text):
    records = []
    seq_id = None
    seq_lines = []
    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]

    for line in lines:
        if line.startswith(">"):
            if seq_id is not None:
                records.append((seq_id, ''.join(seq_lines)))
            seq_id = line[1:].strip()
            seq_lines = []
        else:
            seq_lines.append(line)

    if seq_id is not None:
        records.append((seq_id, ''.join(seq_lines)))

    if not records and lines:
        for idx, seq in enumerate(lines):
            records.append((f"Input_{idx+1}", seq))

    return records

def mutations_to_fasta(df):
    fasta_lines = []
    for _, row in df.iterrows():
        header = f">{row['Mutation']}"
        seq = row["Mutated Peptide"]
        fasta_lines.append(header)
        fasta_lines.append(seq)
    return "\n".join(fasta_lines)

def generate_point_mutations(peptide):
    peptide = peptide.upper()
    variants = []

    for i, original_aa in enumerate(peptide):
        for aa in AMINO_ACIDS:
            if aa != original_aa:
                mutated = list(peptide)
                mutated[i] = aa
                variants.append({
                    "Original Peptide": peptide,
                    "Mutated Peptide": "".join(mutated),
                    "Mutation": f"{original_aa}{i+1}{aa}",
                    "Position": i + 1
                })
    return pd.DataFrame(variants)


def amino_acid_composition(peptide):
    return pd.DataFrame({
        "Amino Acid": AMINO_ACIDS,
        "Count": [peptide.count(aa) for aa in AMINO_ACIDS]
    })


def calculate_charge(peptide):
    pos = peptide.count("K") + peptide.count("R") + peptide.count("H")
    neg = peptide.count("D") + peptide.count("E")
    return pos - neg






# ======================================================
# PAGE 1: ABCP CLASSIFICATION (YOUR CODE)
# ======================================================
if page == "ABCP Classification":

    st.title("SmartABCPeptide")

    st.markdown(
        """
        <div style='background-color: #f0f2f6; border-radius: 10px; padding: 16px; margin-bottom: 16px;color: #333;'>
        <h4>About this tool</h4>
        <p>
        <b>SmartABCPeptide</b> is a machine learning-based web application designed for the classification
        of anticancer peptides (ACPs) and non-ACPs from protein/peptide sequences.
        It supports both single and batch prediction in FASTA or plain sequence format.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_method = st.radio(
        "Input method:",
        ["Manual sequence(s)", "Upload FASTA file"]
    )

    sequence_input = None

    if input_method == "Manual sequence(s)":
        sequence_input = st.text_area(
            "Paste peptide sequence(s) (FASTA or one-per-line)",
            height=150
        )
    else:
        fasta_file = st.file_uploader("Upload a FASTA file", type=["fa", "fasta"])
        if fasta_file:
            sequence_input = fasta_file.getvalue().decode("utf-8")

    if st.button("Classify"):
        if not sequence_input or not sequence_input.strip():
            st.warning("Please provide at least one peptide sequence.")
        else:
            try:
                records = parse_fasta(sequence_input)
                results = []

                for seq_id, seq in records:
                    pred_class, pred_prob = classify_peptide(f">{seq_id}\n{seq}")
                    results.append({
                        "Identifier": seq_id,
                        "Sequence": seq,
                        "Predicted Class": pred_class,
                        "Predicted Probability": f"{pred_prob:.4f}"
                    })

                df = pd.DataFrame(results)
                st.success("Classification results")
                st.dataframe(df)

                csv = df.to_csv(index=False)
                st.download_button(
                    "Download results as CSV",
                    csv,
                    "abcp_classification_results.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error(f"Error during classification:\n{e}")

# ======================================================
# PAGE 2: PEPTIDE DESIGN
# ======================================================
elif page == "Peptide Design":

    st.title("🔁 Peptide Design – Point Mutation Library")

    peptide = st.text_input(
        "Enter peptide sequence",
        placeholder="ACDEFGHIK"
    )

    if peptide:
        peptide = peptide.upper()
        invalid = [aa for aa in peptide if aa not in AMINO_ACIDS]

        if invalid:
            st.error(f"Invalid amino acids found: {set(invalid)}")
        else:
            df = generate_point_mutations(peptide)

            st.success(
                f"Generated {len(df)} variants "
                f"({len(peptide)} positions × 19 mutations)"
            )

            st.dataframe(df, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "Download mutation library (CSV)",
                    df.to_csv(index=False),
                    "point_mutation_library.csv",
                    "text/csv"
                )

            with col2:
                fasta_data = mutations_to_fasta(df)
                st.download_button(
                    "Download mutation library (FASTA)",
                    fasta_data,
                    "point_mutation_library.fasta",
                    "text/plain"
                )




# -------------------------------
# Footer
# -------------------------------
st.markdown(
    """
    <hr>
    <div style='text-align: center; font-size: 15px; color: #666; margin-top: 30px;'>
    Developed by <b>System Biology Laboratory</b><br>
    <b>Indian Institute of Information Technology Allahabad</b>, Prayagraj, India
    </div>
    """,
    unsafe_allow_html=True,
)
