#!/usr/bin/env python
# coding: utf-8
\
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve,
    precision_score, recall_score, f1_score
)
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean, cosine
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import lightly
from lightly.data.multi_view_collate import MultiViewCollate
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import Dataset, DataLoader, random_split # Import necessary classes
import gc
import os
import tempfile
import warnings
from datetime import datetime
import time

multiomics_df = pd.read_csv('final_clp_multiomics.csv')
#multiomics_df = pd.read_csv('final_clp_multiomics_eQTL_based.csv')
print("Successfully loaded clp_multiomics.csv")
print(f"Dataset shape: {multiomics_df.shape}")
print("\nFirst 5 rows:\n", multiomics_df.head())


# Check value distributions for key columns
if 'Cleft_Status' in multiomics_df.columns:
    print("\nCLP_Status distribution:")
    print(multiomics_df['Cleft_Status'].value_counts(normalize=True))


# In[90]:


if 'Zygosity' in multiomics_df.columns:
    print("\nZygosity distribution:")
    print(multiomics_df['Zygosity'].value_counts())


# In[96]:


# Set aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# 1. Twin Distribution by Zygosity
plt.figure(figsize=(8, 6))
if 'Zygosity' in multiomics_df.columns:
    zygosity_counts = multiomics_df['Zygosity'].value_counts()
    ax = sns.barplot(x=zygosity_counts.index, y=zygosity_counts.values, palette="Blues_d")
    plt.title('Distribution of Twins by Zygosity')
    plt.xlabel('Zygosity Type')
    plt.ylabel('Count')
    for i, v in enumerate(zygosity_counts.values):
        ax.text(i, v + 0.1, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('twin_zygosity_distribution.png', dpi=300)
    plt.show()


# In[98]:


# 2. CLP Status Distribution
plt.figure(figsize=(8, 6))
if 'Cleft_Status' in multiomics_df.columns:
    clp_counts = multiomics_df['Cleft_Status'].value_counts()
    colors = ['#ff9999','#66b3ff']
    ax = sns.barplot(x=clp_counts.index, y=clp_counts.values, palette=colors)
    plt.title('Distribution of CLP Status in Twin Dataset')
    plt.xlabel('Cleft Status')
    plt.ylabel('Count')
    
    # Add percentage labels
    total = sum(clp_counts.values)
    for i, v in enumerate(clp_counts.values):
        ax.text(i, v + 0.1, f"{v} ({v/total:.1%})", ha='center')
    
    plt.tight_layout()
    plt.savefig('clp_status_distribution.png', dpi=300)
    plt.show()


# In[217]:


all_cols = multiomics_df.columns.tolist()
# Identify columns by prefix
expression_cols = [col for col in all_cols if col.startswith('Gene_')]
genotype_cols = [col for col in all_cols if col.startswith('Snp_')]
environment_cols = [col for col in all_cols if col.startswith('Covt_')]
target_col = 'Cleft_Status' # Your generated status column name
id_cols = ['SampleID', 'TwinID', 'PairID', 'FamilyID', 'IndividualID', 'Zygosity'] # List of your ID/non-numeric/structure cols


# In[101]:


#selected_gene_cols = random.sample(gene_cols, 10)
selected_gene_cols = expression_cols[20:40]

#selected_snp_cols = random.sample(snp_cols, 10)
selected_snp_cols = genotype_cols[20:40]


selected_features = selected_gene_cols + selected_snp_cols + environment_cols
selected_features.append(target_col)

features_df = multiomics_df[selected_features]
numeric_features_df = features_df.select_dtypes(include=[np.number]) 


if numeric_features_df.shape[1] < 2:
    print("Not enough numeric features selected to draw a correlation matrix.")
else:
    corr_matrix = numeric_features_df.corr()

    # Use the mask to show only the lower triangle (or upper, your code used upper)
    # Using upper triangle mask as in your original code
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(max(10, len(numeric_features_df.columns)*0.5), max(8, len(numeric_features_df.columns)*0.4))) # Adjust size dynamically
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=False, fmt='.2f') # annot=False due to potentially many features

    plt.title('Correlation Matrix of Selected Individual-Level Features')
    plt.tight_layout()
    plt.savefig('selected_features_correlation_matrix.png', dpi=300)
    plt.show()

# --- Note on Other Correlation Types ---
print("\nNote: This heatmap shows individual-level correlations for a subset of features.")
print("Other relevant correlations in twin studies include:")
print("1. Within-pair correlations (e.g., Gene_X in Twin1 vs. Gene_X in Twin2, separately for MZ and DZ).")
print("2. Correlations between features (or feature differences) and the Pair Phenotype (Concordance/Discordance).")
print("These require different data transformations and visualization approaches.")

# 6. CLP Status by Zygosity
if all(col in multiomics_df.columns for col in ['Cleft_Status', 'Zygosity']):
    plt.figure(figsize=(10, 6))
    cross_tab = pd.crosstab(multiomics_df['Zygosity'], multiomics_df['Cleft_Status'], normalize='index')
    cross_tab.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('CLP Status Distribution by Zygosity')
    plt.xlabel('Zygosity')
    plt.ylabel('Proportion')
    plt.legend(title='CLP Status')
    plt.tight_layout()
    plt.savefig('clp_by_zygosity.png', dpi=300)
    plt.show()


# #### Preprocessing

print(f"Identified views from twin data:")
print(f"Genotype columns: {len(genotype_cols)}")
print(f"Expression columns: {len(expression_cols)}")
print(f"Environment columns: {len(environment_cols)}")
print(f"Target column: {target_col}")

# --- Perform Train/Test Split at the Pair Level ---
RANDOM_STATE = 42
TEST_SIZE = 0.3

# 1. Get unique pairs and their characteristics for stratification
pair_info = multiomics_df.groupby('PairID').agg(
    Zygosity=('Zygosity', 'first'), # Get zygosity for the pair
    num_distinct_statuses=(target_col, 'nunique')
).reset_index()

# Create a stratification key for pairs
# Stratify by Zygosity AND whether the pair is concordant/discordant
pair_info['Stratify_Key'] = pair_info['Zygosity'] + '_' + pair_info['num_distinct_statuses'].astype(str).map({"1": 'Concordant', '2': 'Discordant'})

print(f"\nPair info for splitting:")
print(pair_info['Stratify_Key'].value_counts())

# 2. Perform the split on PairIDs
train_pair_ids, test_pair_ids, _, _ = train_test_split(
    pair_info['PairID'],
    pair_info['Stratify_Key'], # Stratify by the combined key
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=pair_info['Stratify_Key'] # stratify requires the stratification variable here too
)

# 3. Filter the original twin_df based on split PairIDs
train_df = multiomics_df[multiomics_df['PairID'].isin(train_pair_ids)].copy()
test_df = multiomics_df[multiomics_df['PairID'].isin(test_pair_ids)].copy()

print(f"\nSplit data into train/test sets (at Pair Level):")
print(f"Training individuals: {len(train_df)} ({len(train_pair_ids)} pairs)")
print(f"Testing individuals: {len(test_df)} ({len(test_pair_ids)} pairs)")

# Verify pair counts and zygosity distribution in splits
print("\nVerification of split composition:")
print("Train Set Pair Counts:")
print(train_df.groupby('Zygosity')['PairID'].nunique())
print("Test Set Pair Counts:")
print(test_df.groupby('Zygosity')['PairID'].nunique())

#Distinct status Concordance:1, discordance:2
distinct_status_counts = (
    train_df.groupby(['PairID', 'Zygosity'])[target_col]
    .nunique()
    .reset_index()
    .rename(columns={target_col: 'Distinct Statuses in Pair'})
)
pair_distribution = (
    distinct_status_counts
    .groupby(['Zygosity', 'Distinct Statuses in Pair'])
    .size()
    .reset_index(name='Pair Counts')
)

print("Train Set Stratification Key Distribution:")
print(pair_distribution)

# --- Extract Views and Target from Split DataFrames ---

X_genotype_train = train_df[genotype_cols]
X_genotype_test = test_df[genotype_cols]

X_expression_train = train_df[expression_cols]
X_expression_test = test_df[expression_cols]

X_environment_train = train_df[environment_cols]
X_environment_test = test_df[environment_cols]

y_train = train_df[target_col]
y_test = test_df[target_col]

metadata_train = train_df[id_cols].copy()
metadata_test = test_df[id_cols].copy()


# --- Apply StandardScaler to each view (fit on training data only) ---
# Ensure scalers are applied AFTER splitting

genotype_scaler = StandardScaler().fit(X_genotype_train)
X_genotype_train_scaled = genotype_scaler.transform(X_genotype_train)
X_genotype_test_scaled = genotype_scaler.transform(X_genotype_test)

expression_scaler = StandardScaler().fit(X_expression_train)
X_expression_train_scaled = expression_scaler.transform(X_expression_train)
X_expression_test_scaled = expression_scaler.transform(X_expression_test)

environment_scaler = StandardScaler().fit(X_environment_train)
X_environment_train_scaled = environment_scaler.transform(X_environment_train)
X_environment_test_scaled = environment_scaler.transform(X_environment_test)

# Scaled data:
# X_genotype_train_scaled, X_genotype_test_scaled
# X_expression_train_scaled, X_expression_test_scaled
# X_environment_train_scaled, X_environment_test_scaled
# y_train, y_test
# metadata_train, metadata_test

print("\nData splitting and scaling complete.")
print("Shapes of scaled training data:")
print(f"Genotype: {X_genotype_train_scaled.shape}")
print(f"Expression: {X_expression_train_scaled.shape}")
print(f"Environment: {X_environment_train_scaled.shape}")
print(f"Target (y_train): {y_train.shape}")

print("Shapes of scaled testing data:")
print(f"Genotype: {X_genotype_test_scaled.shape}")
print(f"Expression: {X_expression_test_scaled.shape}")
print(f"Environment: {X_environment_test_scaled.shape}")
print(f"Target (y_test): {y_test.shape}")


import random
num_sample_features = 5

# Create a figure with subplots (3 views * 2 columns: Original vs. Scaled)
fig, axes = plt.subplots(3, 2, figsize=(12, 15))

# Flatten axes array for easy iteration
axes = axes.flatten()

# --- Genotype View ---
if X_genotype_train.shape[1] > 0:
    # Select random feature indices
    genotype_sample_indices = random.sample(range(X_genotype_train.shape[1]), min(num_sample_features, X_genotype_train.shape[1]))

    # Plot distributions for selected genotype features
    for i, feature_idx in enumerate(genotype_sample_indices):
        original_data = X_genotype_train.iloc[:, feature_idx]
        scaled_data = X_genotype_train_scaled[:, feature_idx]
        feature_name = X_genotype_train.columns[feature_idx]

        # Plot Original
        sns.histplot(original_data, ax=axes[0], kde=True, color='skyblue', label=f'Original ({feature_name})', stat='density', common_norm=False)
        # Plot Scaled
        sns.histplot(scaled_data, ax=axes[1], kde=True, color='lightcoral', label=f'Scaled ({feature_name})', stat='density', common_norm=False)

    axes[0].set_title('Original Genotype Feature Distributions')
    axes[1].set_title('Scaled Genotype Feature Distributions')
    axes[0].set_xlabel('Original Value')
    axes[1].set_xlabel('Scaled Value (Mean=0, Std Dev=1)')
    axes[0].legend()
    axes[1].legend()
else:
    axes[0].set_title('Genotype View - No Features Available')
    axes[1].set_title('Genotype View - No Features Available')
    axes[0].text(0.5, 0.5, 'No genotype features', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
    axes[1].text(0.5, 0.5, 'No genotype features', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)


# --- Expression View ---
if X_expression_train.shape[1] > 0:
    # Select random feature indices
    expression_sample_indices = random.sample(range(X_expression_train.shape[1]), min(num_sample_features, X_expression_train.shape[1]))

    # Plot distributions for selected expression features
    for i, feature_idx in enumerate(expression_sample_indices):
        original_data = X_expression_train.iloc[:, feature_idx]
        scaled_data = X_expression_train_scaled[:, feature_idx]
        feature_name = X_expression_train.columns[feature_idx]

        # Plot Original
        sns.histplot(original_data, ax=axes[2], kde=True, color='skyblue', label=f'Original ({feature_name})', stat='density', common_norm=False)
        # Plot Scaled
        sns.histplot(scaled_data, ax=axes[3], kde=True, color='lightcoral', label=f'Scaled ({feature_name})', stat='density', common_norm=False)

    axes[2].set_title('Original Expression Feature Distributions')
    axes[3].set_title('Scaled Expression Feature Distributions')
    axes[2].set_xlabel('Original Value')
    axes[3].set_xlabel('Scaled Value (Mean=0, Std Dev=1)')
    axes[2].legend()
    axes[3].legend()
else:
    axes[2].set_title('Expression View - No Features Available')
    axes[3].set_title('Expression View - No Features Available')
    axes[2].text(0.5, 0.5, 'No expression features', horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)
    axes[3].text(0.5, 0.5, 'No expression features', horizontalalignment='center', verticalalignment='center', transform=axes[3].transAxes)


# --- Environment View ---
if X_environment_train.shape[1] > 0:
    # Select random feature indices
    environment_sample_indices = random.sample(range(X_environment_train.shape[1]), min(num_sample_features, X_environment_train.shape[1]))

    # Plot distributions for selected environment features
    for i, feature_idx in enumerate(environment_sample_indices):
        original_data = X_environment_train.iloc[:, feature_idx]
        scaled_data = X_environment_train_scaled[:, feature_idx]
        feature_name = X_environment_train.columns[feature_idx]

        # Plot Original
        sns.histplot(original_data, ax=axes[4], kde=True, color='skyblue', label=f'Original ({feature_name})', stat='density', common_norm=False)
        # Plot Scaled
        sns.histplot(scaled_data, ax=axes[5], kde=True, color='lightcoral', label=f'Scaled ({feature_name})', stat='density', common_norm=False)

    axes[4].set_title('Original Environment Feature Distributions')
    axes[5].set_title('Scaled Environment Feature Distributions')
    axes[4].set_xlabel('Original Value')
    axes[5].set_xlabel('Scaled Value (Mean=0, Std Dev=1)')
    axes[4].legend()
    axes[5].legend()
else:
    axes[4].set_title('Environment View - No Features Available')
    axes[5].set_title('Environment View - No Features Available')
    axes[4].text(0.5, 0.5, 'No environment features', horizontalalignment='center', verticalalignment='center', transform=axes[4].transAxes)
    axes[5].text(0.5, 0.5, 'No environment features', horizontalalignment='center', verticalalignment='center', transform=axes[5].transAxes)


plt.tight_layout()
plt.suptitle('Effect of StandardScaler on Feature Distributions (Sample Features)', y=1.02, fontsize=16) # Add a main title
plt.savefig('scaled_data_distributions.png', dpi=300)



X_train_scaled_combined = pd.concat([
    pd.DataFrame(X_genotype_train_scaled, columns=genotype_cols),
    pd.DataFrame(X_expression_train_scaled, columns=expression_cols),
    pd.DataFrame(X_environment_train_scaled, columns=environment_cols)
], axis=1)

X_test_scaled_combined = pd.concat([
    pd.DataFrame(X_genotype_test_scaled, columns=genotype_cols),
    pd.DataFrame(X_expression_test_scaled, columns=expression_cols),
    pd.DataFrame(X_environment_test_scaled, columns=environment_cols)
], axis=1)


# In[112]:


train_nan_count = X_train_scaled_combined.isnull().sum().sum()
test_nan_count = X_test_scaled_combined.isnull().sum().sum()
print(f"Train set NaN count: {train_nan_count}")
print(f"Test set NaN count: {test_nan_count}")


# In[113]:


# Step 3: Train the logistic regression model
print("\nTraining logistic regression model on combined scaled data...")
baseline_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, solver='liblinear')
baseline_model.fit(X_train_scaled_combined, y_train)


# In[114]:


# Step 4: Evaluate model performance
print("\nEvaluating baseline model performance...")
# Make predictions
y_pred = baseline_model.predict(X_test_scaled_combined)
y_pred_proba = baseline_model.predict_proba(X_test_scaled_combined)[:, 1] # Probability of the positive class (1)


# In[115]:


# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print(f"Baseline Logistic Regression Results (Combined Scaled Data):")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"\nConfusion Matrix:")
print(conf_matrix)
print(f"\nClassification Report:")
print(class_report)


# In[116]:


if hasattr(baseline_model, 'coef_'):
    # Get feature importances from logistic regression coefficients
    # Use the columns from the combined dataframe for feature names
    feature_importance = pd.DataFrame({
        'Feature': X_train_scaled_combined.columns,
        'Importance': np.abs(baseline_model.coef_[0]) # Absolute value of coefficients
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importance (Top 15):") # Display top 15
    print(feature_importance.head(15))

    # Optional: Visualize feature importance (Top N)
    top_n_features_viz = 15 # Number of top features to visualize
    plt.figure(figsize=(10, 8)) # Adjusted figure size for more features
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n_features_viz), palette='viridis')
    plt.title(f'Top {top_n_features_viz} Features in Baseline Logistic Regression Model')
    plt.tight_layout()
    plt.savefig(f'top_{top_n_features_viz}_features_baseline.png', dpi=300)
    plt.show()


# Step 6: ROC curve visualization
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Baseline Logistic Regression (Combined Data)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.6)
plt.savefig("roc_curve_baseline.png", dpi=300)
plt.show()

# Step 7: Save the baseline model results for later comparison
baseline_results = {
    'accuracy': accuracy,
    'auc': auc,
    'confusion_matrix': conf_matrix.tolist(), # Convert numpy array to list for easier saving (e.g., JSON)
    'classification_report': class_report
}


# ## Multi-View Contrastive Learning Model Setup


input_dim_gen = X_genotype_train_scaled.shape[1]
input_dim_exp = X_expression_train_scaled.shape[1]
input_dim_env = X_environment_train_scaled.shape[1]

embedding_dim = 128  # Dimension of the output from the encoder (h in SimCLR)
projection_dim = 64  # Dimension of the output from the projection head (z in SimCLR)
encoder_output_dim = 128 # Corresponds to 'embed_dim' in MultiViewContrastiveModel constructor
projection_output_dim = 64 # Corresponds to 'proj_dim' in MultiViewContrastiveModel constructor

# Define hidden layer dimensions - these are also hyperparameters
hidden_dim_encoder = 256 # Hidden layer size in the MLP encoders (Adjustable)
hidden_dim_proj = 128    # Hidden layer size in the MLP projection heads (Adjustable)
dropout_rate = 0.2      # Dropout rate (Adjustable)

print(f"Model dimensions:")
print(f"  Genotype input dim: {input_dim_gen}")
print(f"  Expression input dim: {input_dim_exp}")
print(f"  Environment input dim: {input_dim_env}")
print(f"  Encoder output dim: {encoder_output_dim}")
print(f"  Projection output dim: {projection_output_dim}")
print(f"  Encoder hidden dim: {hidden_dim_encoder}")
print(f"  Projection hidden dim: {hidden_dim_proj}")
print(f"  Dropout rate: {dropout_rate}")

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        
        # Add a shortcut connection if dimensions don't match
        self.shortcut = None
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        
        self.relu = nn.ReLU()
        
        # Initialize weights with a slightly lower variance for residual connections
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', a=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        
        # Apply shortcut if dimensions don't match
        if self.shortcut is not None:
            residual = self.shortcut(residual)
            
        out += residual  # Add the residual connection
        out = self.relu(out)  # Apply ReLU after addition
        return out


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        
        # Define dimensions with gradual reduction
        dim1 = hidden_dim
        dim2 = hidden_dim // 2
        dim3 = hidden_dim // 3
        
        # Initial projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.BatchNorm1d(dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks with dimension adaptation
        self.res_block1 = ResidualBlock(dim1, dim1 // 2, dim2, dropout_rate)
        self.res_block2 = ResidualBlock(dim2, dim2 // 2, dim3, dropout_rate)
        
        # Output projection
        self.output_projection = nn.Linear(dim3, output_dim)
        
        # Initialize non-residual parts of the network
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and 'res_block' not in name:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d) and 'res_block' not in name:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.output_projection(x)
        return x

class SimCLRProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        
        hidden_dim_proj = hidden_dim // 2
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim_proj),
            nn.BatchNorm1d(hidden_dim_proj),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim_proj, output_dim)
        )
        
    def forward(self, x):
        return self.model(x)

class MultiViewContrastiveModel(nn.Module):
    def __init__(self, dim_gen, dim_exp, dim_env,
                 hidden_dim_encoder,
                 encoder_output_dim,
                 hidden_dim_proj,
                 projection_output_dim,
                 dropout_rate):
        super().__init__()
        
        # Encoders for each view
        self.encoder_gen = MLPEncoder(dim_gen, hidden_dim_encoder, encoder_output_dim, dropout_rate)
        self.encoder_exp = MLPEncoder(dim_exp, hidden_dim_encoder, encoder_output_dim, dropout_rate)
        self.encoder_env = MLPEncoder(dim_env, hidden_dim_encoder, encoder_output_dim, dropout_rate)
        
        # Projection heads
        self.projection_gen = SimCLRProjectionHead(encoder_output_dim, hidden_dim_proj, projection_output_dim, dropout_rate)
        self.projection_exp = SimCLRProjectionHead(encoder_output_dim, hidden_dim_proj, projection_output_dim, dropout_rate)
        self.projection_env = SimCLRProjectionHead(encoder_output_dim, hidden_dim_proj, projection_output_dim, dropout_rate)
        
        # Initialize all weights (this will re-initialize residual blocks too, which is fine)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, view_gen, view_exp, view_env):
        # Get embeddings from encoders (h representations)
        embed_gen = self.encoder_gen(view_gen)
        embed_exp = self.encoder_exp(view_exp)
        embed_env = self.encoder_env(view_env)

        # Get projections for contrastive loss (z representations)
        # Note: Contrastive loss is typically applied in the projection space,
        # while downstream tasks use the embeddings from the encoder output.
        proj_gen = self.projection_gen(embed_gen)
        proj_exp = self.projection_exp(embed_exp)
        proj_env = self.projection_env(embed_env)

        # Return projections for loss calculation, and embeddings for downstream tasks
        return (proj_gen, proj_exp, proj_env), (embed_gen, embed_exp, embed_env)

# Step 5: Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiViewContrastiveModel(
    input_dim_gen,
    input_dim_exp,
    input_dim_env,
    hidden_dim_encoder,
    encoder_output_dim,
    hidden_dim_proj,
    projection_output_dim,
    dropout_rate
).to(device)

print(f"Initialized Multi-View Contrastive Model with architecture parameters and moved to {device}")


# In[179]:


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        # Use cosine similarity - REMOVED FOR MATRIX MULTIPLICATION
        # self.similarity_function = nn.CosineSimilarity(dim=1)
        # CrossEntropyLoss expects logits and class indices (0 to 2*batch_size - 1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        Calculates NT-Xent loss for a batch of positive pairs (z_i, z_j).
        z_i and z_j should be the projected representations of the same sample
        from two different views/augmentations.
        """
        # Ensure inputs are on the same device
        z_j = z_j.to(z_i.device)

        batch_size = z_i.shape[0]

        # # --- Debugging Prints ---
        # print(f"\n--- NTXentLoss Debug ---")
        # print(f"Batch size: {batch_size}")
        # print(f"z_i shape: {z_i.shape}")
        # print(f"z_j shape: {z_j.shape}")
        # # --- End Debugging Prints ---


        # Normalize embeddings
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)
        sim_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        sim_i = torch.matmul(z_i, z_i.T) / self.temperature
        sim_j = torch.matmul(z_j, z_j.T) / self.temperature

        # Create masks to remove self-similarity from negative samples within the same view
        mask_diag = torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)

        # # --- Debugging Prints ---
        # print(f"Debug: Before masked_fill for sim_i")
        # print(f"  sim_i shape: {sim_i.shape}, device: {sim_i.device}")
        # print(f"  mask_diag shape: {mask_diag.shape}, device: {mask_diag.device}")
        # print(f"Debug: Before masked_fill for sim_j")
        # print(f"  sim_j shape: {sim_j.shape}, device: {sim_j.device}")
        # print(f"  mask_diag shape: {mask_diag.shape}, device: {mask_diag.device}") # mask_diag should be the same
  
        sim_i = sim_i.masked_fill(mask_diag, float('-inf'))
        sim_j = sim_j.masked_fill(mask_diag, float('-inf'))

        logits_i = torch.cat([sim_matrix, sim_i], dim=1)
        logits_j = torch.cat([sim_matrix.t(), sim_j], dim=1)

        # Combine logits from both directions
        # Total logits shape: (2 * batch_size, 2 * batch_size)
        logits = torch.cat([logits_i, logits_j], dim=0)

        labels = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        labels = torch.cat([labels, labels], dim=0)


        # Compute cross entropy loss
        loss = self.criterion(logits, labels)

        return loss

# Initialize the loss function
criterion = NTXentLoss(temperature=0.1).to(device)
# Step 7: Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3) 


print("\n--- Dataset and DataLoader Setup ---")

# Step 1: Define custom dataset class for multi-view data
# This class will handle fetching individual samples (one per individual)
class MultiOmicsTwinDataset(Dataset):
    def __init__(self, genotype, expression, environment, labels):
    # Convert DataFrames/Series to numpy arrays if necessary
        # Ensure data is treated as numpy arrays for consistent indexing
        self.genotype = genotype.values if isinstance(genotype, pd.DataFrame) else genotype
        self.expression = expression.values if isinstance(expression, pd.DataFrame) else expression
        self.environment = environment.values if isinstance(environment, pd.DataFrame) else environment
        self.labels = labels.values if isinstance(labels, pd.Series) else labels

        # Convert data to float32 for PyTorch model inputs
        self.genotype = self.genotype.astype(np.float32)
        self.expression = self.expression.astype(np.float32)
        self.environment = self.environment.astype(np.float32)

        # Labels should be long for CrossEntropyLoss
        self.labels = self.labels.astype(np.longlong) # Use longlong for safety

        # Basic validation
        assert len(self.genotype) == len(self.expression) == len(self.environment) == len(self.labels), \
            "All data views and labels must have the same number of samples!"

        print(f"Initialized dataset with {len(self.labels)} samples.")

    def __len__(self):
        # Return the total number of samples (individuals) in the dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Get the data for this index (individual)
        gen = torch.tensor(self.genotype[idx], dtype=torch.float32)
        exp = torch.tensor(self.expression[idx], dtype=torch.float32)
        env = torch.tensor(self.environment[idx], dtype=torch.float32)

        # Get the label for this individual
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Return a tuple of (data, label)
        # data is a tuple of the three views for one individual
        return (gen, exp, env), label

# Step 2: Define a collate function to handle batching
def multi_view_collate_fn(batch):

    # Separate the list of tuples into two lists: one for data (views), one for labels
    data_list, labels_list = zip(*batch)

    gen_list, exp_list, env_list = zip(*data_list)

    # Stack the tensors for each view along the batch dimension (dim=0)
    gen_batch = torch.stack(gen_list, dim=0)
    exp_batch = torch.stack(exp_list, dim=0)
    env_batch = torch.stack(env_list, dim=0)

    # Stack the labels
    labels_batch = torch.stack(labels_list, dim=0)

    # Return the batched views as a tuple and the batched labels
    return (gen_batch, exp_batch, env_batch), labels_batch

# Step 3: Use the already scaled data from the previous step
print("Creating datasets using already scaled data...")

# Create train and test datasets using the previously scaled data
# Assuming X_*_train_scaled and y_train are available numpy arrays or pandas objects
train_dataset = MultiOmicsTwinDataset(
    X_genotype_train_scaled,
    X_expression_train_scaled,
    X_environment_train_scaled,
    y_train # Assuming y_train is the individual-level target
)

test_dataset = MultiOmicsTwinDataset(
    X_genotype_test_scaled,
    X_expression_test_scaled,
    X_environment_test_scaled,
    y_test # Assuming y_test is the individual-level target
)

# Step 4: Create a validation set from the training set
val_size = int(0.2 * len(train_dataset)) # 20% of the training data for validation
train_size = len(train_dataset) - val_size

# Split train dataset into train and validation subsets
# Use random_split for a simple random split
train_subset, val_subset = random_split(
    train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(RANDOM_STATE) # Use random seed for reproducibility
)

print(f"Dataset sizes after splitting train into train/validation:")
print(f"  Training subset (for training loop): {len(train_subset)} samples")
print(f"  Validation subset: {len(val_subset)} samples")
print(f"  Test dataset (original test split): {len(test_dataset)} samples")


# Step 5: Create DataLoaders
# Define batch sizes - these are hyperparameters
train_batch_size = 64
val_batch_size = 64
test_batch_size = 32

# Create DataLoaders using the subsets/datasets and the custom collate function
train_dataloader = DataLoader(
    train_subset, # Use the training subset
    batch_size=train_batch_size,
    shuffle=True, # Shuffle training data
    collate_fn=multi_view_collate_fn, # Use the custom collate function
    num_workers=0, # Set to > 0 for parallel data loading (requires __main__ guard)
    drop_last=True # Drop last batch if incomplete (important for NTXentLoss)
)

val_dataloader = DataLoader(
    val_subset, # Use the validation subset
    batch_size=val_batch_size,
    shuffle=False, 
    collate_fn=multi_view_collate_fn,
    num_workers=0
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=test_batch_size,
    shuffle=False, 
    collate_fn=multi_view_collate_fn,
    num_workers=0
)

print(f"Created DataLoaders with the following number of batches:")
print(f"  Train dataloader: {len(train_dataloader)} batches")
print(f"  Validation dataloader: {len(val_dataloader)} batches")
print(f"  Test dataloader: {len(test_dataloader)} batches")


# Step 6: Inspect a single batch from the train dataloader
print("\nInspecting a single batch from train_dataloader...")
try:
    # Get an iterator from the dataloader
    dataloader_iterator = iter(train_dataloader)
    # Fetch the next batch
    first_batch = next(dataloader_iterator)

    # Inspect the structure
    views, labels = first_batch

    # Check the views (should be tensors)
    gen_batch, exp_batch, env_batch = views
    print(f"  Genotype batch shape: {gen_batch.shape}") # Expected: (batch_size, input_dim_gen)
    print(f"  Expression batch shape: {exp_batch.shape}") # Expected: (batch_size, input_dim_exp)
    print(f"  Environment batch shape: {env_batch.shape}") # Expected: (batch_size, input_dim_env)
    print(f"  Labels shape: {labels.shape}") # Expected: (batch_size,)
    print(f"  Label values (first 5): {labels[:5]}...") # Print first 5 labels
    print(f"  Label dtype: {labels.dtype}") # Should be torch.long

    print("Successfully fetched and inspected one batch.")
except Exception as e:
    print(f"Error inspecting batch: {e}")
    print("Please ensure X_*_train_scaled and y_train variables are correctly defined and available.")

print("\n--- Training Loop Setup ---")

# Step 1: Define training parameters
num_epochs = 50 # Number of training epochs
gradient_clip_val = 1.0 # Value for gradient clipping to prevent exploding gradients
checkpoint_dir = './model_checkpoints' # Directory to save model checkpoints
os.makedirs(checkpoint_dir, exist_ok=True) # Create checkpoint directory if it doesn't exist

best_val_loss = float('inf') # Initialize best validation loss for saving best model
patience = 10  # Early stopping patience: number of epochs to wait for improvement
patience_counter = 0 # Counter for early stopping
log_interval = 10  # Log training progress every N batches


# Step 2: Setup logging
# Create a unique filename for the training log
log_file = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
training_history = [] # List to store epoch-wise training history

print(f"Training will run for {num_epochs} epochs with gradient clipping at {gradient_clip_val}")
print(f"Model checkpoints will be saved to: {checkpoint_dir}")
print(f"Early stopping patience: {patience}")
print(f"Training log will be saved to: {log_file}")

print("\n--- Starting Training ---")
start_time = time.time() # Record start time for total training duration

# Initialize the learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',       # Reduce LR when monitored value stops decreasing
    factor=0.5,       # Multiply LR by this factor when reducing
    patience=3,       # Number of epochs with no improvement after which LR will be reduced
    min_lr=1e-5,      # Lower bound on the learning rate
    verbose=True      # Print message when LR is reduced
)

# Add a warmup schedule for the first few epochs
warmup_epochs = 5
initial_lr = optimizer.param_groups[0]['lr']
warmup_factor = 0.2  # Start with 20% of the target learning rate

for epoch in range(num_epochs):
    epoch_start_time = time.time() # Record start time for the current epoch
    
    # Apply warmup schedule for the first few epochs
    if epoch < warmup_epochs:
        # Linear warmup from warmup_factor*initial_lr to initial_lr
        lr_scale = warmup_factor + (1 - warmup_factor) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr * lr_scale
        print(f"  Warmup: LR set to {optimizer.param_groups[0]['lr']:.6f}")

    # --- Training Phase ---
    model.train() # Set model to training mode
    train_loss = 0.0
    batch_count = 0
    total_grad_norm = 0.0  # Track gradient norms

    print(f"Epoch {epoch+1}/{num_epochs}")

    # Iterate over batches in the training dataloader
    for batch_idx, (views, labels) in enumerate(train_dataloader):
        # Move data from the batch to the specified device (CPU/GPU)
        gen_batch, exp_batch, env_batch = views
        gen_batch = gen_batch.to(device)
        exp_batch = exp_batch.to(device)
        env_batch = env_batch.to(device)

        # Zero the gradients from the previous optimization step
        optimizer.zero_grad()

        try:
            # Forward pass: Get projections and embeddings from the model
            projections, embeddings = model(gen_batch, exp_batch, env_batch)
            proj_gen, proj_exp, proj_env = projections # Projections for contrastive loss

            # Calculate contrastive loss between each pair of view projections for the same sample
            loss_ge = criterion(proj_gen, proj_exp)  # Genotype vs Expression
            loss_gv = criterion(proj_gen, proj_env)  # Genotype vs Environment
            loss_ev = criterion(proj_exp, proj_env)  # Expression vs Environment

            # Combine the losses by taking the average (better for stability)
            loss = (loss_ge + loss_gv + loss_ev) / 3.0

        except Exception as e:
            # Basic error handling for issues during forward/loss calculation
            print(f"Error in forward pass or loss calculation for batch {batch_idx}: {e}")
            # Print shapes for debugging if an error occurs
            print(f"  Batch size: {gen_batch.shape[0]}")
            print(f"  Genotype batch shape: {gen_batch.shape}")
            print(f"  Expression batch shape: {exp_batch.shape}")
            print(f"  Environment batch shape: {env_batch.shape}")
            continue # Skip to the next batch if an error occurs

        # Backward pass: Compute gradients of the loss with respect to model parameters
        loss.backward()

        # Calculate and track gradient norm for monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
        total_grad_norm += grad_norm.item()

        # Optimizer step: Update model parameters using the computed gradients
        optimizer.step()

        # Accumulate the loss for the current epoch
        train_loss += loss.item() # Use .item() to get the scalar value of the loss tensor
        batch_count += 1

        # Log progress every 'log_interval' batches
        if (batch_idx + 1) % log_interval == 0:
            print(f"  Batch {batch_idx+1}/{len(train_dataloader)}: Train Loss = {loss.item():.4f}")

    # Calculate average training loss and gradient norm for the epoch
    avg_train_loss = train_loss / batch_count if batch_count > 0 else float('inf')
    avg_grad_norm = total_grad_norm / batch_count if batch_count > 0 else 0.0

    # --- Validation Phase ---
    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    val_batch_count = 0

    # Disable gradient calculation during validation
    with torch.no_grad():
        # Iterate over batches in the validation dataloader
        for views, labels in val_dataloader:
            # Move data to device
            gen_batch, exp_batch, env_batch = views
            gen_batch = gen_batch.to(device)
            exp_batch = exp_batch.to(device)
            env_batch = env_batch.to(device)

            # Forward pass
            projections, embeddings = model(gen_batch, exp_batch, env_batch)
            proj_gen, proj_exp, proj_env = projections

            # Calculate contrastive loss between all view pairs (same as training)
            loss_ge = criterion(proj_gen, proj_exp)  # Genotype vs Expression
            loss_gv = criterion(proj_gen, proj_env)  # Genotype vs Environment
            loss_ev = criterion(proj_exp, proj_env)  # Expression vs Environment

            # Combine the losses (average)
            loss = (loss_ge + loss_gv + loss_ev) / 3.0

            # Accumulate validation loss
            val_loss += loss.item()
            val_batch_count += 1

    # Calculate average validation loss for the epoch
    avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')

    # Calculate epoch duration
    epoch_time = time.time() - epoch_start_time

    # Print epoch summary
    print(f"Epoch {epoch+1}/{num_epochs} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Validation Loss: {avg_val_loss:.4f}")
    print(f"  Gradient Norm: {avg_grad_norm:.4f}")
    print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"  Time: {epoch_time:.2f} seconds")

    # Update learning rate scheduler based on validation loss
    scheduler.step(avg_val_loss)

    # Save history for plotting later
    training_history.append({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'grad_norm': avg_grad_norm,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'time': epoch_time
    })

    # --- Checkpoint Saving and Early Stopping ---
    # Check if this is the best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0 # Reset patience counter on improvement

        # Save the best model state
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(), # Save model parameters
            'optimizer_state_dict': optimizer.state_dict(), # Save optimizer state
            'scheduler_state_dict': scheduler.state_dict(), # Save scheduler state
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, best_model_path)

        print(f"  New best model saved to {best_model_path}")
    else:
        patience_counter += 1 # Increment patience counter if no improvement
        print(f"  No improvement in validation loss. Patience: {patience_counter}/{patience}")

    # Save checkpoint every N epochs (e.g., every 10 epochs)
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, checkpoint_path)
        print(f"  Checkpoint saved to {checkpoint_path}")

    # Check for early stopping condition
    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation loss.")
        break # Exit the training loop

# Calculate total training time
total_time = time.time() - start_time
print(f"\nTraining completed in {total_time:.2f} seconds")

# Save training history to CSV
history_df = pd.DataFrame(training_history)
history_df.to_csv(log_file, index=False)
print(f"Training history saved to {log_file}")


# Set up the plotting style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (16, 12)
})

# Create a 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Plot Training and Validation Loss
ax1 = axes[0, 0]
ax1.plot(history_df['epoch'], history_df['train_loss'], 'b-', label='Train Loss')
ax1.plot(history_df['epoch'], history_df['val_loss'], 'r-', label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# 2. Plot Learning Rate
ax2 = axes[0, 1]
ax2.plot(history_df['epoch'], history_df['learning_rate'], 'g-')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Learning Rate')
ax2.set_title('Learning Rate Schedule')
ax2.grid(True)

# Use log scale for learning rate
ax2.set_yscale('log')

# 3. Plot Gradient Norm
ax3 = axes[1, 0]
ax3.plot(history_df['epoch'], history_df['grad_norm'], 'm-')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Gradient Norm')
ax3.set_title('Gradient Norm Over Time')
ax3.grid(True)

# 4. Plot Loss with Learning Rate Annotations
ax4 = axes[1, 1]
line1, = ax4.plot(history_df['epoch'], history_df['train_loss'], 'b-', label='Train Loss')
line2, = ax4.plot(history_df['epoch'], history_df['val_loss'], 'r-', label='Validation Loss')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss')
ax4.set_title('Loss with LR Change Points')
ax4.legend()

# Identify learning rate change points
lr_changes = []
for i in range(1, len(history_df)):
    if history_df['learning_rate'].iloc[i] != history_df['learning_rate'].iloc[i-1]:
        lr_changes.append(i)

# Add vertical lines at LR change points
for change_idx in lr_changes:
    epoch = history_df['epoch'].iloc[change_idx]
    lr = history_df['learning_rate'].iloc[change_idx]
    ax4.axvline(x=epoch, color='g', linestyle='--', alpha=0.7)
    ax4.text(epoch + 0.1, max(history_df['train_loss'].max(), history_df['val_loss'].max()) * 0.9, 
             f'LR: {lr:.6f}', rotation=90, color='g')

# Adjust layout for better spacing
plt.tight_layout()

# Optional: Save figure to file
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')

plt.show()

# Create a separate figure for just the loss curves if needed
plt.figure(figsize=(10, 6))
plt.plot(history_df['epoch'], history_df['train_loss'], 'b-', linewidth=2, label='Train Loss')
plt.plot(history_df['epoch'], history_df['val_loss'], 'r-', linewidth=2, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Optional: Save the loss-only figure
plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')

plt.show()


# Add this code after your training loop is complete (after all epochs)
print("\n--- Final Test Evaluation ---")

# Load the best model for evaluation
best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set model to evaluation mode

# Initialize test metrics
test_loss = 0.0
test_batch_count = 0

# Component losses for test set
test_loss_components = {'loss_ge': 0.0, 'loss_gv': 0.0, 'loss_ev': 0.0}

# Evaluate on test data
with torch.no_grad():
    for views, labels in test_dataloader:
        # Move data to device
        gen_batch, exp_batch, env_batch = views
        gen_batch = gen_batch.to(device)
        exp_batch = exp_batch.to(device)
        env_batch = env_batch.to(device)

        # Forward pass
        projections, embeddings = model(gen_batch, exp_batch, env_batch)
        proj_gen, proj_exp, proj_env = projections

        # Calculate contrastive loss between all view pairs
        loss_ge = criterion(proj_gen, proj_exp)  # Genotype vs Expression
        loss_gv = criterion(proj_gen, proj_env)  # Genotype vs Environment
        loss_ev = criterion(proj_exp, proj_env)  # Expression vs Environment

        # Accumulate component losses
        test_loss_components['loss_ge'] += loss_ge.item()
        test_loss_components['loss_gv'] += loss_gv.item()
        test_loss_components['loss_ev'] += loss_ev.item()

        # Combine the losses (average)
        loss = (loss_ge + loss_gv + loss_ev) / 3.0
        
        # Accumulate test loss
        test_loss += loss.item()
        test_batch_count += 1

# Calculate average test loss
avg_test_loss = test_loss / test_batch_count if test_batch_count > 0 else float('inf')

# Calculate average component losses
for component in test_loss_components:
    test_loss_components[component] /= test_batch_count if test_batch_count > 0 else 1.0

# Print test results
print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Component Test Losses:")
print(f"  Genotype-Expression: {test_loss_components['loss_ge']:.4f}")
print(f"  Genotype-Environment: {test_loss_components['loss_gv']:.4f}")
print(f"  Expression-Environment: {test_loss_components['loss_ev']:.4f}")

print(f"Best Validation Loss: {checkpoint['val_loss']:.4f}")
print(f"Best Training Loss: {checkpoint['train_loss']:.4f}")
print(f"Best Model from Epoch: {checkpoint['epoch']}")


########Dataset and DataLoader Visualization#######
print("\n--- Dataset Visualization and Information ---")


dataset_info = pd.DataFrame({
    'Split': ['Train', 'Validation', 'Test'],
    'Samples': [len(train_subset), len(val_subset), len(test_dataset)],
    'Batch Size': [train_batch_size, val_batch_size, test_batch_size],
    'Batches': [len(train_dataloader), len(val_dataloader), len(test_dataloader)]
})
print("Dataset Summary:")
print(dataset_info)


# 2. Visualize class distribution in each split
# Need to extract labels from the subsets and dataset
train_labels = [train_subset[i][1].item() for i in range(len(train_subset))]
val_labels = [val_subset[i][1].item() for i in range(len(val_subset))]
test_labels = [test_dataset[i][1].item() for i in range(len(test_dataset))]


# Create a dataframe for the class distributions
# Use pd.Series to count values, fillna(0) for classes not present in a split
class_dist = pd.DataFrame({
    'Train': pd.Series(train_labels).value_counts(normalize=True).sort_index(),
    'Validation': pd.Series(val_labels).value_counts(normalize=True).sort_index(),
    'Test': pd.Series(test_labels).value_counts(normalize=True).sort_index()
}).fillna(0) * 100  # Convert to percentage and handle missing classes


# Plot class distribution
plt.figure(figsize=(10, 6))
class_dist.plot(kind='bar', ax=plt.gca()) # Use ax=plt.gca() to plot on the current figure
plt.title('Class Distribution Across Dataset Splits')
plt.xlabel('Class')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(title='Split')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300) # Added dpi
plt.close() # Close the figure to free memory

# 3. Visualize feature correlation matrix for each view (using a sample of the full training data)
print("\nVisualizing feature correlations for a sample of the training data...")

# Number of features to sample for correlation visualization (keep manageable)
num_sample_features_corr = 50 # Adjust as needed, but avoid plotting thousands

# Create a figure for the heatmaps
plt.figure(figsize=(18, 6)) # Adjusted figure size

# --- Genotype correlation (Sample) ---
if X_genotype_train_scaled.shape[1] > 0:
    # Sample features if there are too many
    if X_genotype_train_scaled.shape[1] > num_sample_features_corr:
        gen_sample_cols = random.sample(range(X_genotype_train_scaled.shape[1]), num_sample_features_corr)
        gen_sample_data = X_genotype_train_scaled[:, gen_sample_cols]
        gen_sample_col_names = [genotype_cols[i] for i in gen_sample_cols]
    else:
        gen_sample_data = X_genotype_train_scaled
        gen_sample_col_names = genotype_cols

    plt.subplot(1, 3, 1)
    # Calculate correlation matrix for the sampled features
    gen_corr = np.corrcoef(gen_sample_data.T)
    sns.heatmap(gen_corr, cmap='coolwarm', center=0, annot=False, xticklabels=False, yticklabels=False, cbar_kws={'label': 'Correlation'})
    plt.title(f'Genotype Features\nCorrelation (Sample of {gen_sample_data.shape[1]} features)')
else:
    plt.subplot(1, 3, 1)
    plt.title('Genotype Features\nNo Features Available')
    plt.text(0.5, 0.5, 'No genotype features', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)


# --- Expression correlation (Sample) ---
if X_expression_train_scaled.shape[1] > 0:
    # Sample features if there are too many
    if X_expression_train_scaled.shape[1] > num_sample_features_corr:
        exp_sample_cols = random.sample(range(X_expression_train_scaled.shape[1]), num_sample_features_corr)
        exp_sample_data = X_expression_train_scaled[:, exp_sample_cols]
        exp_sample_col_names = [expression_cols[i] for i in exp_sample_cols]
    else:
        exp_sample_data = X_expression_train_scaled
        exp_sample_col_names = expression_cols

    plt.subplot(1, 3, 2)
    # Calculate correlation matrix for the sampled features
    exp_corr = np.corrcoef(exp_sample_data.T)
    sns.heatmap(exp_corr, cmap='coolwarm', center=0, annot=False, xticklabels=False, yticklabels=False, cbar_kws={'label': 'Correlation'})
    plt.title(f'Expression Features\nCorrelation (Sample of {exp_sample_data.shape[1]} features)')
else:
    plt.subplot(1, 3, 2)
    plt.title('Expression Features\nNo Features Available')
    plt.text(0.5, 0.5, 'No expression features', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)


# --- Environment correlation (All or Sample) ---
# Environment usually has fewer features, can often plot all
if X_environment_train_scaled.shape[1] > 0:
     # Decide whether to sample or use all based on the number of env features
    if X_environment_train_scaled.shape[1] > num_sample_features_corr:
        env_sample_cols = random.sample(range(X_environment_train_scaled.shape[1]), num_sample_features_corr)
        env_sample_data = X_environment_train_scaled[:, env_sample_cols]
        env_sample_col_names = [environment_cols[i] for i in env_sample_cols]
        env_plot_title = f'Environment Features\nCorrelation (Sample of {env_sample_data.shape[1]} features)'
        env_annot = False # Turn off annotation for larger samples
    else:
        env_sample_data = X_environment_train_scaled
        env_sample_col_names = environment_cols
        env_plot_title = f'Environment Features\nCorrelation (All {env_sample_data.shape[1]} features)'
        env_annot = (env_sample_data.shape[1] <= 15) # Annotate if few features

    plt.subplot(1, 3, 3)
    # Calculate correlation matrix for the sampled/all features
    env_corr = np.corrcoef(env_sample_data.T)
    sns.heatmap(env_corr, cmap='coolwarm', center=0, annot=env_annot, fmt=".2f",
                xticklabels=(env_sample_col_names if env_annot else False),
                yticklabels=(env_sample_col_names if env_annot else False),
                cbar_kws={'label': 'Correlation'})
    plt.title(env_plot_title)
    if env_annot:
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

else:
    plt.subplot(1, 3, 3)
    plt.title('Environment Features\nNo Features Available')
    plt.text(0.5, 0.5, 'No environment features', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)


plt.tight_layout()
plt.savefig('feature_correlations_sampled.png', dpi=300) # Updated filename
plt.close() # Close the figure


# 4. Print data dimensions summary
print("\nData Dimensions Summary (from scaled training data):")
print(f"  Genotype view: {X_genotype_train_scaled.shape[1]} features")
print(f"  Expression view: {X_expression_train_scaled.shape[1]} features")
print(f"  Environment view: {X_environment_train_scaled.shape[1]} features")
print(f"  Total features: {X_genotype_train_scaled.shape[1] + X_expression_train_scaled.shape[1] + X_environment_train_scaled.shape[1]}")


# 5. Visualize a 2D embedding of samples using PCA (using a sample of the full training data)
print("\nVisualizing PCA of combined features for a sample of the training data...")

# Number of samples to use for PCA visualization (keep manageable, e.g., 1000-5000)
num_sample_pca = min(5000, X_genotype_train_scaled.shape[0]) # Use up to 5000 samples or total if less

# Select random indices for PCA samples from the training set
pca_sample_indices = random.sample(range(X_genotype_train_scaled.shape[0]), num_sample_pca)

# Extract features and labels for the sampled individuals from the full scaled training data
gen_pca_sample = X_genotype_train_scaled[pca_sample_indices, :]
exp_pca_sample = X_expression_train_scaled[pca_sample_indices, :]
env_pca_sample = X_environment_train_scaled[pca_sample_indices, :]
labels_pca_sample = y_train.iloc[pca_sample_indices].values # Get corresponding labels from y_train


# Combine the features for PCA
# Ensure shapes are correct before concatenation (should be num_sample_pca x num_features)
combined_features_pca = np.concatenate([
    gen_pca_sample,
    exp_pca_sample,
    env_pca_sample
], axis=1)


# Apply PCA
pca = PCA(n_components=2)
reduced_features_pca = pca.fit_transform(combined_features_pca)

# Plot PCA
plt.figure(figsize=(8, 6))
# Use a color palette suitable for discrete classes
colors = sns.color_palette('viridis', n_colors=len(np.unique(labels_pca_sample)))
for i, label in enumerate(np.unique(labels_pca_sample)):
    mask = labels_pca_sample == label
    plt.scatter(
        reduced_features_pca[mask, 0],
        reduced_features_pca[mask, 1],
        label=f'Class {label}',
        alpha=0.7,
        s=10, # Adjust point size
        color=colors[i]
    )
plt.title(f'PCA Visualization of Combined Features (Sample of {num_sample_pca} individuals)')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)') # Add explained variance
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)') # Add explained variance
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data_pca_visualization_sampled.png', dpi=300) # Updated filename
plt.close() # Close the figure


print("\n--- Dataset Visualization Complete ---")


# --- Embedding Generation ---
print("\n--- Embedding Generation ---")

# Step 1: Load the best model from checkpoints
best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
if os.path.exists(best_model_path):
    # Load checkpoint
    checkpoint = torch.load(best_model_path, weights_only=True)  # Use weights_only=True for security
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}")
else:
    print("Warning: No checkpoint found. Using the current model state.")

# Set model to evaluation mode
model.eval()


print("\nScaling full dataset with memory-efficient approach...")

def transform_and_save_in_chunks(data_df, scaler, output_file, chunk_size=100, dtype=np.float32):
    """Transform data in tiny chunks and immediately save to disk."""
    n_samples = len(data_df)
    n_features = len(data_df.columns)
    n_chunks = int(np.ceil(n_samples / chunk_size))
    
    # Create memory-mapped array for output
    mmap = np.memmap(output_file, dtype=dtype, mode='w+', 
                     shape=(n_samples, n_features))
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_samples)
        
        # Get a small chunk
        chunk_df = data_df.iloc[start_idx:end_idx]
        
        # Transform and immediately write to disk
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            transformed_chunk = scaler.transform(chunk_df)
        
        # Save to memory-mapped file
        mmap[start_idx:end_idx] = transformed_chunk.astype(dtype)
        
        # Force clear memory
        del chunk_df, transformed_chunk
        gc.collect()
        
        # Print progress
        if (i + 1) % 10 == 0 or (i + 1) == n_chunks:
            print(f"  Processed chunk {i+1}/{n_chunks} ({end_idx}/{n_samples} samples)")
    
    # Flush changes to disk
    mmap.flush()
    del mmap
    gc.collect()
    
    return output_file

# Create temporary directory for memory-mapped files
temp_dir = tempfile.mkdtemp()
print(f"Created temporary directory for memory-mapped files: {temp_dir}")

try:
    # Process one modality at a time, clearing memory between each
    
    # 1. Genotype data
    print("Scaling genotype data...")
    genotype_file = os.path.join(temp_dir, "genotype_scaled.dat")
    transform_and_save_in_chunks(
        multiomics_df[genotype_cols], 
        genotype_scaler, 
        genotype_file
    )
    # Clear original data from memory
    gc.collect()
    print("Genotype data scaled and saved to disk.")
    
    # 2. Expression data (likely the largest)
    print("Scaling expression data...")
    # Use an even smaller chunk size for expression data
    expression_file = os.path.join(temp_dir, "expression_scaled.dat")
    transform_and_save_in_chunks(
        multiomics_df[expression_cols], 
        expression_scaler, 
        expression_file,
        chunk_size=50  # Smaller chunk size for expression data
    )
    # Clear original data
    gc.collect()
    print("Expression data scaled and saved to disk.")
    
    # 3. Environment data
    print("Scaling environment data...")
    environment_file = os.path.join(temp_dir, "environment_scaled.dat")
    transform_and_save_in_chunks(
        multiomics_df[environment_cols], 
        environment_scaler, 
        environment_file
    )
    gc.collect()
    print("Environment data scaled and saved to disk.")
    
    # Now load the memory-mapped arrays for use (these don't fully load into RAM)
    print("Creating memory-mapped arrays for scaled data...")
    X_genotype_scaled = np.memmap(
        genotype_file, 
        dtype=np.float32, 
        mode='r', 
        shape=(len(multiomics_df), len(genotype_cols))
    )
    
    X_expression_scaled = np.memmap(
        expression_file, 
        dtype=np.float32, 
        mode='r', 
        shape=(len(multiomics_df), len(expression_cols))
    )
    
    X_environment_scaled = np.memmap(
        environment_file, 
        dtype=np.float32, 
        mode='r', 
        shape=(len(multiomics_df), len(environment_cols))
    )
    
    print("Memory-mapped arrays created. Full dataset scaling complete.")
    
    # Now you can use these X_*_scaled arrays in your existing code
    # They act like numpy arrays but don't consume much memory
    
    # Optional: Convert to regular numpy arrays if needed 
    # (only do this if you have enough RAM!)
    # X_genotype_scaled_array = np.array(X_genotype_scaled)
    # X_expression_scaled_array = np.array(X_expression_scaled)
    # X_environment_scaled_array = np.array(X_environment_scaled)
    
except Exception as e:
    print(f"Error during scaling: {e}")
    import traceback
    traceback.print_exc()

def cleanup_temp_files():
    """Clean up temporary memory-mapped files when done"""
    try:
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        os.rmdir(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Error during cleanup: {e}")


y = multiomics_df[target_col]
# Step 2: Create a dataset containing ALL twin samples
full_dataset = MultiOmicsTwinDataset(
    X_genotype_scaled,
    X_expression_scaled, 
    X_environment_scaled,
    y
)

# Create a dataloader with no shuffling to maintain sample order
full_dataloader = DataLoader(
    full_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=multi_view_collate_fn,
    num_workers=0
)

print(f"Created full dataset with {len(full_dataset)} twin samples")


all_gen_embeddings = []
all_exp_embeddings = []
all_env_embeddings = []
all_combined_embeddings = []
all_indices = []
sample_labels = []

# Track batch indices for alignment
current_idx = 0

with torch.no_grad():
    for views, labels in full_dataloader:
        # Move data to device
        gen_batch, exp_batch, env_batch = views
        gen_batch = gen_batch.to(device)
        exp_batch = exp_batch.to(device)
        env_batch = env_batch.to(device)
        
        # Forward pass to get embeddings
        _, embeddings = model(gen_batch, exp_batch, env_batch)
        embed_gen, embed_exp, embed_env = embeddings
        
        # Convert to numpy and store
        gen_np = embed_gen.cpu().numpy()
        exp_np = embed_exp.cpu().numpy()
        env_np = embed_env.cpu().numpy()
        
        # Store embeddings
        all_gen_embeddings.append(gen_np)
        all_exp_embeddings.append(exp_np)
        all_env_embeddings.append(env_np)
        
        # Create combined embedding (average of all views)
        combined_embed = (gen_np + exp_np + env_np) / 3.0
        all_combined_embeddings.append(combined_embed)
        
        # Track indices for alignment
        batch_size = gen_batch.shape[0]
        batch_indices = list(range(current_idx, current_idx + batch_size))
        all_indices.extend(batch_indices)
        current_idx += batch_size

# Concatenate all batches
all_gen_embeddings = np.vstack(all_gen_embeddings)
all_exp_embeddings = np.vstack(all_exp_embeddings)
all_env_embeddings = np.vstack(all_env_embeddings)
all_combined_embeddings = np.vstack(all_combined_embeddings)
sample_labels = np.array(y)

print(f"Generated embeddings for {len(all_indices)} samples")
print(f"Embedding dimensions - Genotype: {all_gen_embeddings.shape}, Expression: {all_exp_embeddings.shape}, Environment: {all_env_embeddings.shape}")
print(f"Combined embedding shape: {all_combined_embeddings.shape}")

# Step 4: Ensure alignment with metadata
# Instead of trying to add to an existing DataFrame, create a fresh one
embedding_df = pd.DataFrame()

# Add embedding data one row at a time to avoid dimensionality issues
embedding_data = []
for i in range(len(sample_labels)):
    embedding_data.append({
        'class_label': sample_labels[i],
        'genotype_embedding': all_gen_embeddings[i],
        'expression_embedding': all_exp_embeddings[i],
        'environment_embedding': all_env_embeddings[i],
        'combined_embedding': all_combined_embeddings[i]
    })

# Create DataFrame from embedding data
embedding_df = pd.DataFrame(embedding_data)

# Add family and zygosity information if available
if 'FamilyID' in multiomics_df.columns and 'Zygosity' in multiomics_df.columns:
    # Create a mapping dictionary for quick lookup
    # This assumes there's a one-to-one mapping between indices in the embedding_df
    # and rows in the twin_df
    if len(embedding_df) == len(multiomics_df):
        embedding_df['FamilyID'] = multiomics_df['FamilyID'].values
        embedding_df['Zygosity'] = multiomics_df['Zygosity'].values
        embedding_df['PairID'] = multiomics_df['PairID'].values
        print("Added family and zygosity metadata to embeddings")
    else:
        print(f"Warning: Length mismatch between embedding_df ({len(embedding_df)}) and twin_df ({len(twin_df)})")
        print("Skipping family and zygosity metadata addition")

print("\nEmbedding DataFrame created with the following columns:")
print(embedding_df.columns.tolist())
print(f"Shape: {embedding_df.shape}")

try:
    # Check if embedding_df exists in memory
    embedding_df
    print(f"Using in-memory embeddings with shape: {embedding_df.shape}")
except NameError:
    # Load from saved file
    embeddings_path = 'twin_embeddings.pkl'
    embedding_df = pd.read_pickle(embeddings_path)
    print(f"Loaded embeddings from disk with shape: {embedding_df.shape}")

embedding_train_df = embedding_df[embedding_df['PairID'].isin(train_pair_ids)].copy()
embedding_test_df = embedding_df[embedding_df['PairID'].isin(test_pair_ids)].copy()


# Stack both training and test embeddings consistently
X_train = np.stack(embedding_train_df['combined_embedding'])
X_test = np.stack(embedding_test_df['combined_embedding'])

# Get labels
y_train = embedding_train_df['class_label']
y_test = embedding_test_df['class_label']

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Step 3: Train a logistic regression model on the embeddings
# Using the same settings as baseline for fair comparison
print("\nTraining logistic regression model on embeddings...")
embedding_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, solver='liblinear')
embedding_model.fit(X_train, y_train)

# Step 4: Evaluate model performance
print("\nEvaluating embedding-based model performance...")
# Make predictions
y_pred = embedding_model.predict(X_test)
y_pred_proba = embedding_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print(f"Embedding-based Logistic Regression Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"\nConfusion Matrix:")
print(conf_matrix)
print(f"\nClassification Report:")
print(class_report)

plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Embedding-based Logistic Regression')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.6)
plt.savefig("roc_curve_embedding.png", dpi=300)
plt.show()

# Step 5: Load baseline results (from your previously saved baseline model)
try:
    # If you saved the baseline results as a dictionary or file
    baseline_accuracy = baseline_results['accuracy']
    baseline_auc = baseline_results['auc']
    
    # Compare with baseline
    print("\nComparison with Baseline Model:")
    print(f"                   Raw Features    Embeddings")
    print(f"Accuracy:          {baseline_accuracy:.4f}        {accuracy:.4f}")
    print(f"AUC:               {baseline_auc:.4f}        {auc:.4f}")
    
    # Calculate improvement
    acc_improvement = (accuracy - baseline_accuracy) / baseline_accuracy * 100
    auc_improvement = (auc - baseline_auc) / baseline_auc * 100
    
    print(f"Improvement:       Accuracy: {acc_improvement:.2f}%  |  AUC: {auc_improvement:.2f}%")
except Exception as e:
    print(f"\nCould not load baseline results for comparison: {str(e)}")
    print("To compare models, run the baseline model first and store its results.")


# --- Evaluation: Twin Similarity Analysis ---
print("\n--- Evaluation: Twin Similarity Analysis ---")
# Step 1: Load twin data with zygosity information
try:
    # Check if embedding_df exists and has zygosity information
    if 'embedding_df' in locals() and 'Zygosity' in embedding_df.columns:
        print(f"Using in-memory embeddings with zygosity data")
    else:
        # Load from saved file
        embeddings_path = 'twin_embeddings.pkl'
        embedding_df = pd.read_pickle(embeddings_path)
        print(f"Loaded embeddings from disk with shape: {embedding_df.shape}")
        
        if 'Zygosity' not in embedding_df.columns:
            print("Error: No zygosity information found in the embeddings dataframe")
            raise ValueError("Missing zygosity data")
except Exception as e:
    print(f"Error loading twin data: {str(e)}")
    raise

# Step 2: Group twins by family ID
if 'FamilyID' not in embedding_df.columns:
    print("Error: No family ID information found in the embeddings dataframe")
    raise ValueError("Missing family ID data")

print(f"Found {embedding_df['FamilyID'].nunique()} twin families in the dataset")

# Step 3: Calculate embedding distances between twin pairs
twin_distances = []

# Group twins by family
family_groups = embedding_df.groupby('FamilyID')

# Calculate distances for each pair
for family_id, family_data in family_groups:
    # Skip families with only one member
    if len(family_data) < 2:
        continue
        
    # Get zygosity (should be the same for all members of the family)
    zygosity = family_data['Zygosity'].iloc[0]
    
    # Get embeddings for this family
    family_embeddings = np.vstack(family_data['combined_embedding'].values)
    
    # Calculate pairwise distances within family
    # For each possible pair in the family
    for i in range(len(family_data)):
        for j in range(i+1, len(family_data)):
            # Get embeddings for twin pair
            embed1 = family_data['combined_embedding'].iloc[i]
            embed2 = family_data['combined_embedding'].iloc[j]
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(embed1 - embed2)
            
            # Store results
            twin_distances.append({
                'FamilyID': family_id,
                'Zygosity': zygosity,
                'Distance': distance
            })

# Convert to DataFrame
distance_df = pd.DataFrame(twin_distances)
print(f"Calculated distances for {len(distance_df)} twin pairs")

# Step 4: Compare distances between MZ and DZ twins
# Split by zygosity
try:
    mz_distances = distance_df[distance_df['Zygosity'] == 'MZ']['Distance'].values
    dz_distances = distance_df[distance_df['Zygosity'] == 'DZ']['Distance'].values
    
    print(f"Monozygotic (MZ) twin pairs: {len(mz_distances)}")
    print(f"Dizygotic (DZ) twin pairs: {len(dz_distances)}")
    
    # Calculate summary statistics
    mz_mean = np.mean(mz_distances)
    dz_mean = np.mean(dz_distances)
    mz_std = np.std(mz_distances)
    dz_std = np.std(dz_distances)
    
    print(f"MZ twins mean distance: {mz_mean:.4f} ± {mz_std:.4f}")
    print(f"DZ twins mean distance: {dz_mean:.4f} ± {dz_std:.4f}")
    
    # Step 5: Statistical testing on distance differences
    # Perform t-test to compare means
    t_stat, p_value = stats.ttest_ind(mz_distances, dz_distances, equal_var=False)
    
    print("\nStatistical Analysis:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("  Result: Significant difference between MZ and DZ twin similarity")
    else:
        print("  Result: No significant difference between MZ and DZ twin similarity")
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(mz_distances) - 1) * mz_std**2 + 
                          (len(dz_distances) - 1) * dz_std**2) / 
                         (len(mz_distances) + len(dz_distances) - 2))
    
    cohens_d = np.abs(mz_mean - dz_mean) / pooled_std
    
    print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
    
    # Step 6: Visualize the results
    plt.figure(figsize=(12, 6))
    
    # Violin plots
    plt.subplot(1, 2, 1)
    sns.violinplot(x='Zygosity', y='Distance', data=distance_df, inner='box', palette='Set2')
    plt.title('Embedding Distances by Zygosity')
    plt.ylabel('Euclidean Distance')
    
    # Add mean lines
    plt.axhline(mz_mean, color='blue', linestyle='--', alpha=0.5, label=f'MZ Mean: {mz_mean:.4f}')
    plt.axhline(dz_mean, color='green', linestyle='--', alpha=0.5, label=f'DZ Mean: {dz_mean:.4f}')
    plt.legend()
    
    # Histograms
    plt.subplot(1, 2, 2)
    sns.histplot(mz_distances, color='blue', alpha=0.5, label='MZ Twins', kde=True)
    sns.histplot(dz_distances, color='green', alpha=0.5, label='DZ Twins', kde=True)
    plt.title('Distribution of Twin Pair Distances')
    plt.xlabel('Euclidean Distance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('twin_similarity_analysis.png')
    plt.close()
    
    # Bonus: Box plots with individual points
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Zygosity', y='Distance', data=distance_df, palette='Set2')
    sns.stripplot(x='Zygosity', y='Distance', data=distance_df, 
                 size=4, color='.3', alpha=0.5)
    
    plt.title('Twin Pair Distances by Zygosity')
    plt.ylabel('Euclidean Distance in Embedding Space')
    
    # Add p-value annotation
    if p_value < 0.001:
        p_text = 'p < 0.001'
    else:
        p_text = f'p = {p_value:.3f}'
        
    plt.text(0.5, plt.ylim()[1] * 0.9, p_text, 
             horizontalalignment='center', size='large')
    
    plt.tight_layout()
    plt.savefig('twin_distances_boxplot.png')
    plt.close()
    
except Exception as e:
    print(f"Error in twin similarity analysis: {str(e)}")

print("\n--- Twin Similarity Analysis Complete ---")

# --- Evaluation: Twin Similarity Analysis on Raw Data ---
print("\n--- Evaluation: Twin Similarity Analysis on Raw Data ---")

# Combine all features for a complete raw data representation
all_feature_cols = genotype_cols + expression_cols + environment_cols

# Step 4: Calculate distances between twin pairs using raw features
raw_twin_distances = []

# Group twins by family
family_groups = multiomics_df.groupby('FamilyID')

# Calculate distances for each pair
for family_id, family_data in family_groups:
    # Skip families with only one member
    if len(family_data) < 2:
        continue
        
    # Get zygosity (should be the same for all members of the family)
    zygosity = family_data['Zygosity'].iloc[0]
    
    # Calculate pairwise distances within family
    # For each possible pair in the family
    for i in range(len(family_data)):
        for j in range(i+1, len(family_data)):
            # Get raw features for twin pair
            features1 = family_data[all_feature_cols].iloc[i].values
            features2 = family_data[all_feature_cols].iloc[j].values
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(features1 - features2)
            
            # Store results
            raw_twin_distances.append({
                'FamilyID': family_id,
                'Zygosity': zygosity,
                'Distance': distance
            })

# Convert to DataFrame
raw_distance_df = pd.DataFrame(raw_twin_distances)
print(f"Calculated raw feature distances for {len(raw_distance_df)} twin pairs")

# Step 5: Compare distances between MZ and DZ twins using raw features
# Split by zygosity
try:
    raw_mz_distances = raw_distance_df[raw_distance_df['Zygosity'] == 'MZ']['Distance'].values
    raw_dz_distances = raw_distance_df[raw_distance_df['Zygosity'] == 'DZ']['Distance'].values
    
    print(f"Monozygotic (MZ) twin pairs: {len(raw_mz_distances)}")
    print(f"Dizygotic (DZ) twin pairs: {len(raw_dz_distances)}")
    
    # Calculate summary statistics
    raw_mz_mean = np.mean(raw_mz_distances)
    raw_dz_mean = np.mean(raw_dz_distances)
    raw_mz_std = np.std(raw_mz_distances)
    raw_dz_std = np.std(raw_dz_distances)
    
    print(f"RAW DATA - MZ twins mean distance: {raw_mz_mean:.4f} ± {raw_mz_std:.4f}")
    print(f"RAW DATA - DZ twins mean distance: {raw_dz_mean:.4f} ± {raw_dz_std:.4f}")
    
    # Step 6: Statistical testing on distance differences
    # Perform t-test to compare means
    raw_t_stat, raw_p_value = stats.ttest_ind(raw_mz_distances, raw_dz_distances, equal_var=False)
    
    print("\nStatistical Analysis (Raw Features):")
    print(f"  t-statistic: {raw_t_stat:.4f}")
    print(f"  p-value: {raw_p_value:.6f}")
    
    if raw_p_value < 0.05:
        print("  Result: Significant difference between MZ and DZ twin similarity using raw features")
    else:
        print("  Result: No significant difference between MZ and DZ twin similarity using raw features")
    
    # Calculate effect size (Cohen's d)
    raw_pooled_std = np.sqrt(((len(raw_mz_distances) - 1) * raw_mz_std**2 + 
                             (len(raw_dz_distances) - 1) * raw_dz_std**2) / 
                            (len(raw_mz_distances) + len(raw_dz_distances) - 2))
    
    raw_cohens_d = np.abs(raw_mz_mean - raw_dz_mean) / raw_pooled_std
    
    print(f"  Effect size (Cohen's d): {raw_cohens_d:.4f}")
    
    # Step 7: Visualize the results
    plt.figure(figsize=(12, 6))
    
    # Violin plots
    plt.subplot(1, 2, 1)
    sns.violinplot(x='Zygosity', y='Distance', data=raw_distance_df, inner='box', palette='Set2')
    plt.title('Raw Feature Distances by Zygosity')
    plt.ylabel('Euclidean Distance')
    
    # Add mean lines
    plt.axhline(raw_mz_mean, color='blue', linestyle='--', alpha=0.5, label=f'MZ Mean: {raw_mz_mean:.4f}')
    plt.axhline(raw_dz_mean, color='green', linestyle='--', alpha=0.5, label=f'DZ Mean: {raw_dz_mean:.4f}')
    plt.legend()
    
    # Histograms
    plt.subplot(1, 2, 2)
    sns.histplot(raw_mz_distances, color='blue', alpha=0.5, label='MZ Twins', kde=True)
    sns.histplot(raw_dz_distances, color='green', alpha=0.5, label='DZ Twins', kde=True)
    plt.title('Distribution of Twin Pair Distances (Raw Features)')
    plt.xlabel('Euclidean Distance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('raw_twin_similarity_analysis.png')
    plt.close()
    
    # Bonus: Box plots with individual points
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Zygosity', y='Distance', data=raw_distance_df, palette='Set2')
    sns.stripplot(x='Zygosity', y='Distance', data=raw_distance_df, 
                 size=4, color='.3', alpha=0.5)
    
    plt.title('Twin Pair Distances by Zygosity (Raw Features)')
    plt.ylabel('Euclidean Distance in Raw Feature Space')
    
    # Add p-value annotation
    if raw_p_value < 0.001:
        p_text = 'p < 0.001'
    else:
        p_text = f'p = {raw_p_value:.3f}'
        
    plt.text(0.5, plt.ylim()[1] * 0.9, p_text, 
             horizontalalignment='center', size='large')
    
    plt.tight_layout()
    plt.savefig('raw_twin_distances_boxplot.png')
    plt.close()
    
    # Step 8: Compare raw feature distances with embedding distances (if available)
    try:
        # Try to load the embedding distance data if not already in memory
        if 'distance_df' not in locals():
            try:
                # Check if we can load from disk
                distance_df = pd.read_csv('embedding_twin_distances.csv')
                print("Loaded embedding distance data from disk")
            except:
                print("Could not load embedding distance data for comparison")
                raise ValueError("Missing embedding distance data")
        
        # Get embedding distance statistics
        emb_mz_distances = distance_df[distance_df['Zygosity'] == 'MZ']['Distance'].values
        emb_dz_distances = distance_df[distance_df['Zygosity'] == 'DZ']['Distance'].values
        
        emb_mz_mean = np.mean(emb_mz_distances)
        emb_dz_mean = np.mean(emb_dz_distances)
        
        # Calculate difference ratios (how much better is MZ vs DZ separation)
        raw_diff_ratio = raw_dz_mean / raw_mz_mean
        emb_diff_ratio = emb_dz_mean / emb_mz_mean
        
        print("\nComparison of Raw Features vs Embeddings:")
        print(f"  Raw features - MZ/DZ distance ratio: {raw_diff_ratio:.4f}")
        print(f"  Embeddings - MZ/DZ distance ratio: {emb_diff_ratio:.4f}")
        
        improvement = (emb_diff_ratio / raw_diff_ratio - 1) * 100
        print(f"  Improvement in MZ/DZ separation: {improvement:.2f}%")
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        # Set up data for grouped bar plot
        data = pd.DataFrame({
            'Data Type': ['Raw Features', 'Raw Features', 'Embeddings', 'Embeddings'],
            'Zygosity': ['MZ', 'DZ', 'MZ', 'DZ'],
            'Mean Distance': [raw_mz_mean, raw_dz_mean, emb_mz_mean, emb_dz_mean],
            'Std': [raw_mz_std, raw_dz_std, np.std(emb_mz_distances), np.std(emb_dz_distances)]
        })
        
        # Create grouped bar plot
        plt.subplot(1, 2, 1)
        sns.barplot(x='Data Type', y='Mean Distance', hue='Zygosity', data=data)
        plt.title('Mean Twin Distances: Raw vs Embeddings')
        plt.ylabel('Mean Euclidean Distance')
        
        # Create comparison of effect sizes
        plt.subplot(1, 2, 2)
        effect_sizes = pd.DataFrame({
            'Data Type': ['Raw Features', 'Embeddings'],
            "Cohen's d": [raw_cohens_d, cohens_d]
        })
        
        sns.barplot(x='Data Type', y="Cohen's d", data=effect_sizes)
        plt.title("Effect Size Comparison")
        plt.ylabel("Cohen's d (MZ vs DZ Separation)")
        
        plt.tight_layout()
        plt.savefig('raw_vs_embedding_comparison.png')
        plt.close()
        
    except Exception as e:
        print(f"Could not complete raw vs embedding comparison: {str(e)}")
    
except Exception as e:
    print(f"Error in raw data twin similarity analysis: {str(e)}")

print("\n--- Raw Data Twin Similarity Analysis Complete ---")

