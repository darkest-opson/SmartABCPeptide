# app.py
import streamlit as st
import pandas as pd
from abcp_classification_function import classify_peptide

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="SmartABCPeptide",
    layout="centered"
)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# -------------------------------
# Custom CSS (INTERACTIVE UI)
# -------------------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #2c3e50;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 20px;
}

.card {
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.footer {
    text-align: center;
    padding: 15px;
    font-size: 14px;
    color: #777;
}
</style>
""", unsafe_allow_html=True)

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


def generate_point_mutations(peptide):
    peptide = peptide.upper()
    variants = []

    for i, original_aa in enumerate(peptide):
        for aa in AMINO_ACIDS:
            if aa != original_aa:
                mutated = list(peptide)
                mutated[i] = aa
                variants.append({
                    "Mutated Peptide": "".join(mutated),
                    "Mutation": f"{original_aa}{i+1}{aa}",
                    "Position": i + 1
                })
    return pd.DataFrame(variants)


def calculate_charge(peptide):
    pos = peptide.count("K") + peptide.count("R") + peptide.count("H")
    neg = peptide.count("D") + peptide.count("E")
    return pos - neg


# ======================================================
# HEADER
# ======================================================
st.markdown('<div class="main-title">SmartABCPeptide</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Anti-Breast Cancer Peptide Classification & Design Platform</div>', unsafe_allow_html=True)

# ======================================================
# SECTION 1: CLASSIFICATION
# ======================================================
st.markdown('<div class="card">', unsafe_allow_html=True)

st.header("ABCP Classification")

input_method = st.radio("Input method", ["Manual Input", "Upload FASTA"])

sequence_input = None

if input_method == "Manual Input":
    sequence_input = st.text_area("Enter sequence(s)", height=120)
else:
    fasta_file = st.file_uploader("Upload FASTA", type=["fa", "fasta"])
    if fasta_file:
        sequence_input = fasta_file.getvalue().decode("utf-8")

if st.button("Run Classification"):
    if not sequence_input:
        st.warning("Please provide sequence input")
    else:
        with st.spinner("Analyzing peptides..."):
            records = parse_fasta(sequence_input)
            results = []

            for seq_id, seq in records:
                pred_class, prob = classify_peptide(f">{seq_id}\n{seq}")

                results.append({
                    "ID": seq_id,
                    "Sequence": seq,
                    "Class": pred_class,
                    "Probability": round(prob, 4),
                    "Charge": calculate_charge(seq),
                    "Length": len(seq)
                })

            df = pd.DataFrame(results)

            st.success("Classification Complete")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "⬇ Download CSV",
                df.to_csv(index=False),
                "classification_results.csv"
            )

st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# SECTION 2: PEPTIDE DESIGN + PREDICTION
# ======================================================
st.markdown('<div class="card">', unsafe_allow_html=True)

st.header("Peptide Design + Prediction")

peptide = st.text_input("Enter peptide sequence")

top_n = st.slider("Select number of top peptides", 5, 50, 10)

if peptide:
    peptide = peptide.upper()

    invalid = [aa for aa in peptide if aa not in AMINO_ACIDS]

    if invalid:
        st.error(f"Invalid amino acids: {set(invalid)}")
    else:
        st.info(f"Total mutations to evaluate: {len(peptide) * 19}")

        if st.button("Generate & Predict"):

            with st.spinner("Generating mutations and predicting..."):

                df_mut = generate_point_mutations(peptide)
                predictions = []

                for _, row in df_mut.iterrows():
                    seq = row["Mutated Peptide"]

                    pred_class, prob = classify_peptide(f">mut\n{seq}")

                    predictions.append({
                        "Mutation": row["Mutation"],
                        "Sequence": seq,
                        "Position": row["Position"],
                        "Class": pred_class,
                        "Probability": round(float(prob), 4),
                        "Charge": calculate_charge(seq)
                    })

                df_pred = pd.DataFrame(predictions)

                # Sort by best candidates
                df_pred = df_pred.sort_values(by="Probability", ascending=False)

                st.success("Mutation + Prediction Completed")

                # 🔥 TOP CANDIDATES
                st.subheader(f" Top {top_n} Peptides")
                st.dataframe(df_pred.head(top_n), use_container_width=True)

                # 📊 FULL DATA
                with st.expander("View All Mutations"):
                    st.dataframe(df_pred, use_container_width=True)


                # ⬇ DOWNLOADS
                col1, col2 = st.columns(2)

                with col1:
                    st.download_button(
                        "⬇ Download Top Peptides",
                        df_pred.head(top_n).to_csv(index=False),
                        "top_peptides.csv"
                    )

                with col2:
                    st.download_button(
                        "⬇ Download All Results",
                        df_pred.to_csv(index=False),
                        "all_mutations_predictions.csv"
                    )

st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# SECTION 3: ABOUT
# ======================================================
st.markdown('<div class="card">', unsafe_allow_html=True)

# st.header("About")

st.markdown("""
<div style='text-align: center;'>

<h3>Developers</h3>
<p>
Prabhat Tripathi, Sana Tarannum, Devesh Somvanshi, Sankalp Patil,<br>
Srishti Chakraborty, Ankish Arya, Pritish Varadwaj, Nirmalya Sen
</p>

<h3>Institutions</h3>
<p>
Indian Institute of Information Technology Allahabad (IIIT-A), Prayagraj<br>
Bose Institute, Kolkata
</p>

<h3>Citation</h3>
<p>
Tripathi et al. (2026). SmartABCPeptide: ML-based ACP prediction tool.
</p>

</div>
""", unsafe_allow_html=True)

# ======================================================
# FOOTER (PROFESSIONAL)
# ======================================================
st.markdown("""
<hr>
<div class="footer">
Developed by <b>System Biology Laboratory</b><br>
Indian Institute of Information Technology Allahabad<br><br>
© 2026 SmartABCPeptide | All Rights Reserved
</div>
""", unsafe_allow_html=True)