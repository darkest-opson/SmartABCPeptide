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

# Load pre-trained models and scalers
gnb = joblib.load("gnb_model.pkl")
rf_model = joblib.load("rf_model.pkl")
scaler_rf = joblib.load("scaler_rf.pkl")
ab_model = joblib.load("adaboost_model.pkl")
scaler_ab = joblib.load("scaler_ab.pkl")
et_model = joblib.load("extratrees_model.pkl")
scaler_et = joblib.load("scaler_et.pkl")

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

# Load the ANN model
model_path = "ann_model.pth"
input_dim = 100
ann_model = EnhancedANN(input_dim)
ann_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
ann_model.eval()

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, input_channels, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Load the CNN model
model_path = "cnn_model.pth"
input_channels = 1
input_dim = 100
cnn_model = CNN(input_channels, input_dim)
cnn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
cnn_model.eval()

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
            command = f"python ./iFeature/iFeature-master/iFeature.py --file {input_file} --type {feature_type} --out {csv_output_file}"
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
        headers_to_keep = ['SPS', 'FpDensityMorgan1', 'FpDensityMorgan2', 'BCUT2D_MWLOW', 'BCUT2D_CHGLO', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'Chi1', 'Chi2n', 'Chi2v', 'Kappa2', 'Kappa3', 'PEOE_VSA6', 'PEOE_VSA7', 'SMR_VSA4', 'SMR_VSA5', 'SlogP_VSA1', 'SlogP_VSA4', 'SlogP_VSA5', 'EState_VSA10', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA9', 'VSA_EState2', 'VSA_EState4', 'FractionCSP3', 'NumRotatableBonds', 'MolLogP', 'fr_NH2', 'A', 'C.1', 'D', 'E', 'F', 'G', 'H.1', 'I', 'K', 'L', 'M', 'N.1', 'P', 'Q', 'R', 'S.1', 'T', 'V', 'W', 'Y', 'hydrophobicity_PRAM900101.G1', 'hydrophobicity_PRAM900101.G2', 'hydrophobicity_PONP930101.G1', 'hydrophobicity_PONP930101.G3', 'hydrophobicity_CASG920101.G3', 'normwaalsvolume.G1', 'solventaccess.G3', 'hydrophobicity_PRAM900101.3.residue75', 'hydrophobicity_ARGP820101.3.residue0', 'hydrophobicity_ZIMJ680101.1.residue50', 'hydrophobicity_ZIMJ680101.1.residue75', 'hydrophobicity_ZIMJ680101.1.residue100', 'hydrophobicity_PONP930101.2.residue25', 'hydrophobicity_PONP930101.3.residue25', 'hydrophobicity_PONP930101.3.residue50', 'hydrophobicity_PONP930101.3.residue75', 'hydrophobicity_PONP930101.3.residue100', 'hydrophobicity_CASG920101.3.residue0', 'hydrophobicity_CASG920101.3.residue25', 'hydrophobicity_CASG920101.3.residue50', 'hydrophobicity_CASG920101.3.residue100', 'hydrophobicity_ENGD860101.2.residue100', 'hydrophobicity_ENGD860101.3.residue0', 'hydrophobicity_ENGD860101.3.residue50', 'hydrophobicity_ENGD860101.3.residue75', 'hydrophobicity_FASG890101.3.residue0', 'polarity.1.residue75', 'polarity.1.residue100', 'polarizability.1.residue0', 'charge.1.residue75', 'charge.2.residue75', 'secondarystruct.2.residue75', 'secondarystruct.3.residue75', 'solventaccess.1.residue50', 'solventaccess.1.residue75', 'solventaccess.2.residue0', 'hydrophobicity_PRAM900101.Tr1331', 'hydrophobicity_PRAM900101.Tr2332', 'hydrophobicity_ARGP820101.Tr2332', 'hydrophobicity_PONP930101.Tr1331', 'hydrophobicity_PONP930101.Tr2332', 'hydrophobicity_CASG920101.Tr1221', 'hydrophobicity_CASG920101.Tr2332', 'hydrophobicity_ENGD860101.Tr2332', 'hydrophobicity_FASG890101.Tr1331', 'normwaalsvolume.Tr1331', 'normwaalsvolume.Tr2332', 'polarity.Tr1331', 'polarizability.Tr2332', 'secondarystruct.Tr1221']

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

    def ensemble_predict(df):
        # Convert DataFrame to NumPy array to avoid feature name warnings
        X = df.values

        # GNB predictions
        gnb_prob = gnb.predict_proba(X)[:, 1]
        gnb_pred = gnb.predict(X)

        # RF predictions
        X_rf = scaler_rf.transform(X)  # Use NumPy array
        rf_prob = rf_model.predict_proba(X_rf)[:, 1]
        rf_pred = rf_model.predict(X_rf)

        # AdaBoost predictions
        X_ab = scaler_ab.transform(X)  # Use NumPy array
        ab_prob = ab_model.predict_proba(X_ab)[:, 1]
        ab_pred = ab_model.predict(X_ab)

        # ExtraTrees predictions
        X_et = scaler_et.transform(X)  # Use NumPy array
        et_prob = et_model.predict_proba(X_et)[:, 1]
        et_pred = et_model.predict(X_et)

        # ANN predictions
        X_ann = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            ann_prob = ann_model(X_ann).cpu().numpy().flatten()
        ann_pred = (ann_prob > 0.5).astype(int)

        # CNN predictions
        X_cnn = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            cnn_prob = cnn_model(X_cnn).cpu().numpy().flatten()
        cnn_pred = (cnn_prob > 0.5).astype(int)

        # Ensemble predictions
        predictions = np.vstack([ann_pred, cnn_pred, gnb_pred, rf_pred, ab_pred, et_pred])
        final_pred, _ = mode(predictions, axis=0, keepdims=True)
        final_pred = final_pred.flatten()

        # Ensemble probabilities
        probabilities = np.vstack([ann_prob, cnn_prob, gnb_prob, rf_prob, ab_prob, et_prob])
        probabilities = np.clip(probabilities, 1e-5, 1 - 1e-5)
        final_prob = np.mean(probabilities, axis=0)

        # Prepare results
        results = []
        for i in range(len(final_pred)):
            prob_abcp = final_prob[i]
            label = "ABCP" if final_pred[i] == 1 else "NON-ABCP"
            results.append((label, prob_abcp))
            print(results)

            # Print simplified results
            print(f"Class: {label}, Probability: {round(prob_abcp, 4)}")
        return results # Return only the first result (for single sequence input)
    # Main execution flow
    fasta_file = convert_to_fasta(sequence)
    feature_df1 = extract_ifeatures(fasta_file)
    feature_df2 = extract_peptide_descriptors(fasta_file)
    feature_df3 = generate_protparam_features(fasta_file)
    feature_df4 = calculate_atomic_composition(fasta_file)

    final_df = pd.concat([feature_df1, feature_df2, feature_df3, feature_df4], axis=1)
    output_file = "peptide_combinedfeatures.xlsx"
    final_df.to_excel(output_file, index=False)

    filtered_df = filter_columns_by_headers(output_file)
    if filtered_df is not None:
        filtered_df.to_excel("filtered_peptide_features.xlsx", index=False)
        result = ensemble_predict(filtered_df)
    
    return result[0]
