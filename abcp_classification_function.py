import subprocess
import pandas as pd
import os
from Bio import SeqIO
from rdkit.Chem import Descriptors, MolFromSmiles
from rdkit import Chem
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
from scipy.stats import mode
import joblib
import torch
import torch.nn as nn
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# Define the EnhancedANN model
class EnhancedANN(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


rf = joblib.load("rf_model.pkl")
svm = joblib.load("svm_model.pkl")
mlp = joblib.load("mlp_model.pkl")
ab = joblib.load("adaboost_model.pkl")

# Load shared scaler
scaler = joblib.load("scaler_mlp.pkl")  # You can reuse this for all

# Load ANN model
ann = EnhancedANN(input_dim=150)  # use your correct input dimension
ann.load_state_dict(torch.load("ann_model.pth", map_location="cpu"))
ann.eval()

# Combine into model_dict
model_dict = {
    "RF": rf,
    "SVM": svm,
    "MLP": mlp,
    "AB": ab,
    "ANN": ann
}


# Function to classify peptide
def classify_peptide(sequence):
    def convert_to_fasta(sequence, filename="peptide.fasta", header=">sequence"):
        lines = sequence.strip().split("\n")
        if lines[0].startswith(">"):
            fasta_content = sequence.strip()
        else:
            fasta_content = f"{header}\n{sequence.strip()}"
        with open(filename, "w") as fasta_file:
            fasta_file.write(fasta_content)
        return filename

    def extract_ifeatures(input_file):
        feature_types = ['AAC', 'DPC', 'CTDC', 'CTDD', 'CTDT']
        feature_dfs = []

        for feature_type in feature_types:
            csv_output_file = f"{os.path.splitext(input_file)[0]}_{feature_type.lower()}.csv"
            command = f"python iFeature/iFeature-master/iFeature.py --file {input_file} --type {feature_type} --out {csv_output_file}"
            subprocess.run(command, shell=True, check=True)

            try:
                feature_df = pd.read_csv(csv_output_file, header=0, index_col=False, sep="\t")
            except pd.errors.ParserError:
                feature_df = pd.read_csv(csv_output_file, header=0, index_col=False, sep=",")

            feature_dfs.append(feature_df)
            os.remove(csv_output_file)

        final_df = feature_dfs[0]
        for df in feature_dfs[1:]:
            final_df = pd.merge(final_df, df, how='inner', left_on=final_df.columns[0], right_on=df.columns[0])

        final_df = final_df.drop(columns=[final_df.columns[0]])
        return final_df

    def extract_peptide_descriptors(input_fasta):
        sequences = [str(record.seq) for record in SeqIO.parse(input_fasta, "fasta")]
        all_descriptors = pd.DataFrame()

        for sequence in sequences:
            try:
                smiles_string = Chem.MolToSmiles(Chem.MolFromSequence(sequence))
                mol = MolFromSmiles(smiles_string)

                desc_values = {desc_name: descriptor(mol) if mol else None
                               for desc_name, descriptor in Descriptors._descList}
                all_descriptors = pd.concat([all_descriptors, pd.DataFrame(desc_values, index=[0])], ignore_index=True)
            except Exception as e:
                print(f"Error processing sequence {sequence}: {e}")

        return all_descriptors

    def generate_protparam_features(fasta_file):
        peptides = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
        results = []

        for peptide in peptides:
            analysis = ProteinAnalysis(peptide)
            results.append({
                'Number of Amino Acids': len(peptide),
                'Molecular Weight': analysis.molecular_weight(),
                'Aromaticity': analysis.aromaticity(),
                'GRAVY': analysis.gravy(),
                'Isoelectric Point': analysis.isoelectric_point(),
                'Charge at pH 7': analysis.charge_at_pH(pH=7),
                'Alpha-Helix Fraction': analysis.secondary_structure_fraction()[0],
                'Beta-Sheet Fraction': analysis.secondary_structure_fraction()[2],
                'Coil Fraction': analysis.secondary_structure_fraction()[1],
                'Molar Extinction Coefficient (Reduced Cysteines)': analysis.molar_extinction_coefficient()[0],
                'Molar Extinction Coefficient (Oxidized Cysteines)': analysis.molar_extinction_coefficient()[1]
            })

        return pd.DataFrame(results)

    def calculate_atomic_composition(fasta_file):
        def atomic_composition(protein):
            atomic_weights = {
                'A': (3, 7, 2, 1, 0), 'R': (6, 14, 2, 4, 0), 'N': (4, 8, 3, 2, 0), 'D': (4, 7, 4, 1, 0),
                'C': (3, 7, 2, 1, 1), 'Q': (5, 10, 3, 2, 0), 'E': (5, 9, 4, 1, 0), 'G': (2, 5, 2, 1, 0),
                'H': (6, 9, 2, 3, 0), 'I': (6, 13, 2, 1, 0), 'L': (6, 13, 2, 1, 0), 'K': (6, 14, 2, 2, 0),
                'M': (5, 11, 2, 1, 1), 'F': (9, 11, 2, 1, 0), 'P': (5, 9, 2, 1, 0), 'S': (3, 7, 3, 1, 0),
                'T': (4, 9, 3, 1, 0), 'W': (11, 12, 2, 2, 0), 'Y': (9, 11, 3, 1, 0), 'V': (5, 11, 2, 1, 0)
            }
            c = h = o = n = s = 0
            for aa in protein:
                if aa in atomic_weights:
                    c_add, h_add, o_add, n_add, s_add = atomic_weights[aa]
                    c += c_add; h += h_add; o += o_add; n += n_add; s += s_add
            return c, h, o, n, s

        results = []
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequence = str(record.seq)
            composition = atomic_composition(sequence)
            results.append(list(composition))

        return pd.DataFrame(results, columns=["C", "H", "O", "N", "S"])

    def filter_columns_by_headers(file_path):
        headers_to_keep = ['SMR_VSA4', 'MaxPartialCharge', 'BCUT2D_MWHI', 'Kappa3', 'charge.2.residue0', 'hydrophobicity_PRAM900101.3.residue0', 'SlogP_VSA4', 'hydrophobicity_CASG920101.2.residue0', 'hydrophobicity_CASG920101.3.residue75', 'hydrophobicity_ARGP820101.Tr2332', 'solventaccess.1.residue0', 'secondarystruct.1.residue0', 'PEOE_VSA6', 'hydrophobicity_ARGP820101.3.residue25', 'hydrophobicity_CASG920101.1.residue0', 'hydrophobicity_CASG920101.3.residue0', 'fr_NH1', 'solventaccess.3.residue0', 'MinPartialCharge', 'NumAtomStereoCenters', 'polarizability.3.residue25', 'charge.Tr1221', 'normwaalsvolume.3.residue0', 'hydrophobicity_ARGP820101.1.residue25', 'polarizability.3.residue0', 'normwaalsvolume.3.residue25', 'normwaalsvolume.3.residue75', 'secondarystruct.2.residue50', 'secondarystruct.3.residue100', 'hydrophobicity_ARGP820101.3.residue0', 'MaxAbsPartialCharge', 'secondarystruct.1.residue50', 'hydrophobicity_FASG890101.3.residue50', 'polarizability.3.residue75', 'hydrophobicity_FASG890101.1.residue50', 'charge.1.residue75', 'EState_VSA4', 'secondarystruct.1.residue75', 'hydrophobicity_PONP930101.3.residue75', 'polarizability.Tr1221', 'normwaalsvolume.1.residue25', 'polarity.1.residue75', 'hydrophobicity_PRAM900101.3.residue25', 'hydrophobicity_ZIMJ680101.1.residue75', 'hydrophobicity_CASG920101.3.residue100', 'hydrophobicity_PRAM900101.1.residue0', 'hydrophobicity_FASG890101.1.residue0', 'charge.2.residue25', 'KK', 'hydrophobicity_FASG890101.1.residue75', 'hydrophobicity_ARGP820101.3.residue50', 'secondarystruct.2.residue100', 'charge.1.residue100', 'secondarystruct.3.residue0', 'hydrophobicity_ENGD860101.3.residue75', 'fr_C_O_noCOO', 'hydrophobicity_ZIMJ680101.3.residue0', 'solventaccess.3.residue75', 'hydrophobicity_FASG890101.3.residue25', 'normwaalsvolume.1.residue75', 'hydrophobicity_CASG920101.1.residue25', 'hydrophobicity_PRAM900101.2.residue0', 'hydrophobicity_PRAM900101.3.residue50', 'hydrophobicity_PRAM900101.1.residue75', 'VSA_EState8', 'NumAmideBonds', 'hydrophobicity_CASG920101.2.residue50', 'hydrophobicity_PONP930101.1.residue50', 'charge.1.residue25', 'hydrophobicity_ENGD860101.1.residue50', 'hydrophobicity_ARGP820101.2.residue50', 'hydrophobicity_ZIMJ680101.1.residue50', 'fr_amide', 'solventaccess.3.residue50', 'hydrophobicity_PONP930101.1.residue75', 'hydrophobicity_CASG920101.2.residue25', 'hydrophobicity_FASG890101.1.residue100', 'secondarystruct.1.residue100', 'hydrophobicity_PRAM900101.2.residue75', 'charge.1.residue0', 'hydrophobicity_CASG920101.1.residue50', 'hydrophobicity_PONP930101.2.residue75', 'normwaalsvolume.2.residue75', 'normwaalsvolume.2.residue25', 'Molar Extinction Coefficient (Oxidized Cysteines)', 'qed', 'BCUT2D_MWLOW', 'hydrophobicity_ZIMJ680101.2.residue100', 'secondarystruct.3.residue50', 'M', 'polarizability.1.residue50', 'G', 'hydrophobicity_ZIMJ680101.2.residue75', 'hydrophobicity_ENGD860101.2.residue50', 'hydrophobicity_ARGP820101.3.residue100', 'PEOE_VSA4', 'hydrophobicity_ARGP820101.2.residue100', 'polarizability.2.residue50', 'hydrophobicity_PRAM900101.2.residue100', 'normwaalsvolume.1.residue50', 'fr_NH2', 'hydrophobicity_ENGD860101.Tr2332', 'W', 'hydrophobicity_CASG920101.2.residue100', 'BCUT2D_CHGLO', 'KL', 'polarity.Tr2332', 'WK', 'fr_benzene', 'hydrophobicity_ARGP820101.Tr1331', 'charge.2.residue100', 'LF', 'BCUT2D_LOGPLOW', 'NumSaturatedHeterocycles', 'MaxEStateIndex', 'MaxAbsEStateIndex', 'KI', 'S.1', 'NumAromaticCarbocycles', 'fr_Ar_N', 'AF', 'fr_phenol_noOrthoHbond', 'fr_guanido', 'fr_COO', 'ST', 'LK', 'WR', 'CT', 'KF', 'GL', 'TY', 'fr_COO2', 'GH', 'GR', 'fr_Al_OH_noTert', 'SC', 'NumAromaticRings', 'HN', 'QL', 'ED', 'FG', 'CN', 'fr_Al_COO', 'PR', 'FW', 'fr_phenol', 'RP', 'TI', 'GD', 'TC']
        if not os.path.exists(file_path):
            print(f"Error: File not found at '{file_path}'.")
            return None

        df = pd.read_excel(file_path)
        existing_headers = [col for col in headers_to_keep if col in df.columns]
        if not existing_headers:
            print("Warning: No matching columns found.")
            return None

        filtered_df = df[existing_headers]
        return filtered_df


    def ensemble_soft_vote_predict(df_features, scaler, model_dict):
        X = df_features.values
        X_scaled = scaler.transform(df_features.values)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        X_tensor_cnn = X_tensor.reshape(X_tensor.shape[0], 1, X_tensor.shape[1])

        # Dictionary to store the probabilities and classes for each model
        probs_dict = {
            "RF": model_dict["RF"].predict_proba(X_scaled)[:, 1],
            "SVM": model_dict["SVM"].predict_proba(X_scaled)[:, 1],
            "MLP": model_dict["MLP"].predict_proba(X_scaled)[:, 1],
            "AB": model_dict["AB"].predict_proba(X_scaled)[:, 1], # Added GNB
            "ANN": model_dict["ANN"](X_tensor).detach().numpy().squeeze()
        }

        # Ensure all probabilities are scalar (not arrays)
        for model_name, prob in probs_dict.items():
            if isinstance(prob, np.ndarray) and prob.size == 1:  # If it's an array with a single value
                probs_dict[model_name] = prob.item()  # Convert to a scalar
            elif isinstance(prob, np.ndarray):  # If it's a multi-element array
                probs_dict[model_name] = prob[0]  # Take the first element

        # Calculate the average probability across all models
        avg_prob = np.mean(list(probs_dict.values()), axis=0)

        # Store predicted classes and probabilities based on the average probability
        predicted_class_label = np.where(avg_prob > 0.5, "ABCP", "NON-ABCP")
        predicted_class_prob = np.where(avg_prob > 0.5, avg_prob, 1 - avg_prob)

        # Construct the results DataFrame
        results = pd.DataFrame({
            "Predicted Class": predicted_class_label,
            "Predicted Probability": predicted_class_prob
        }, index=df_features.index)

        return results

# Main execution flow
    fasta_file = convert_to_fasta(sequence)
    feature_df1 = extract_ifeatures(fasta_file)
    feature_df2 = extract_peptide_descriptors(fasta_file)
    feature_df3 = generate_protparam_features(fasta_file)
    feature_df4 = calculate_atomic_composition(fasta_file)

    final_df = pd.concat([feature_df1, feature_df2, feature_df3, feature_df4], axis=1)
    unknown_data = "peptide_combinedfeatures.xlsx"
    final_df.to_excel(unknown_data, index=False)

    filtered_df = filter_columns_by_headers(unknown_data)
    if filtered_df is not None:
        filtered_df.to_excel("filtered_peptide_features.xlsx", index=False)
        result_df = ensemble_soft_vote_predict(filtered_df, scaler, model_dict)

    # print(f"Predicted Class: {result_df['Predicted Class'].iloc[0]}")
    # print(f"Predicted Probability: {result_df['Predicted Probability'].iloc[0]:.6f}")
    return str(result_df['Predicted Class'].iloc[0]), float(result_df['Predicted Probability'].iloc[0])

