import numpy as np
import pandas as pd
import itertools
import subprocess
from copula_tools import CopulaScaler
from pomegranate import BayesianNetwork
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata

def ipf_in(target, source, columns, path):
    target_ipf = target.copy()
    source_ipf = source.copy()
    if "WIF" in target_ipf:  
        target_ipf["WIF"] += 1
        source_ipf["WIF"] += 1
    source_ipf.to_csv(f"{path}/source.csv", index=False)
    for col in columns:
        unique, counts = np.unique(target_ipf[col], return_counts=True)
        # Remove values from target that are not in source
        unique_source = np.unique(source_ipf[col])
        unique_cleaned = list(unique)
        counts_cleaned = list(counts)
        for value in unique:
            if value not in unique_source:
                idx = unique_cleaned.index(value)
                unique_cleaned.pop(idx)
                counts_cleaned.pop(idx)
        df = pd.DataFrame(dict(zip(unique_cleaned, counts_cleaned)), index=[0])
        # Put 0 where data in source not in target
        for value in unique_source:
            if value not in unique:
                df[value] = 0
        df.to_csv(f"{path}/{col}.csv", index=False)


def sample_ipf(path):
    # Run R ipf script
    subprocess.call(
        ["Rscript", f"{path}/ipf.R"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    # Generate samples from weights
    ipf_w = pd.read_csv(f"{path}/weights.csv")
    ipf_w["weight"] = (ipf_w["weight"]).astype(int)
    sample = ipf_w.loc[ipf_w.index.repeat(ipf_w["weight"])].reset_index(drop=True)
    sample.drop(["weight", "id", "geo_all", "avg_weight" ,"weight_factor"], axis=1, inplace=True)
    if "WIF" in sample: sample["WIF"] -= 1
    return sample


def sample_copula(source, target, sampler, sampler_args={}):
    source_scaler = CopulaScaler()
    target_scaler = CopulaScaler()
    source_scaler.fit(source)
    target_scaler.fit(target)
    source_tr = source_scaler.transform(source)
    sample = sampler(source_tr, len(target), **sampler_args)
    sample = source_scaler.interpolation(sample, source.columns)
    return target_scaler.inverse_transform(sample)


def sample_bn(source, n, n_jobs=1):
    bn = BayesianNetwork.from_samples(source, algorithm="greedy", n_jobs=n_jobs)
    sample = pd.DataFrame(bn.sample(n, algorithm="rejection"), columns=source.columns)
    return sample


def sample_ctgan(source, n, ctgan_args={}):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=source)
    ctgan = CTGANSynthesizer(metadata, **ctgan_args)
    ctgan.fit(source)
    return ctgan.sample(num_rows=n)


def sample_tvae(source, n, tvae_args={}):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=source)
    ctgan = TVAESynthesizer(metadata, **tvae_args)
    ctgan.fit(source)
    return ctgan.sample(num_rows=n)


def sample_independent(source, target):
    columns = source.columns
    # Independent baseline
    ind_data = np.zeros(shape=target.shape)
    for i in range(ind_data.shape[0]):
        for j in range(ind_data.shape[1]):
            ind_data[i,j] = source[columns[j]].sample(1)
    return pd.DataFrame(ind_data, columns=columns)


def srmse(data1, data2):
    """ Compute Standardized Root Mean Squared Error between two datasets.

    Reference: https://www.researchgate.net/publication/282815687_A_Bayesian_network_approach_for_population_synthesis
    """
    columns = list(data1.columns.values)
    # Relative frequency
    data1_f = data1.value_counts(normalize=True)
    data2_f = data2.value_counts(normalize=True)
    # Total numbers of categories
    Mi = [data1_f.index.get_level_values(l).union(data2_f.index.get_level_values(l)).unique().size for l in range(len(columns))]
    M = np.prod(Mi)
    # SRMSE
    SRMSE = ((data1_f.subtract(data2_f, fill_value=0)**2) * M).sum()**(.5)
    return SRMSE


def sampling_zeros(source, target, synthetic):
    """Count the combinations of variables from the synthetic data which 
    are in the test set but not in the training set.

    Reference: https://arxiv.org/pdf/1909.07689.pdf
    """
    source_set = set(tuple(i) for i in source.to_numpy())
    target_set = set(tuple(i) for i in target.to_numpy())
    synthetic_set = set(tuple(i) for i in synthetic.to_numpy())
    zeros = synthetic_set.intersection(target_set) - source_set
    return len(zeros)


def result_table(source, target, synthetic_data, columns, max_projection=5, save=False, path=""):
    # Calculate SRMSE and zeros
    srmse_dict = {}
    for model in synthetic_data:
        srmse_dict[model] = {}
        df = synthetic_data[model]
        for i in range(1, max_projection+1):
            tuples = list(itertools.combinations(columns, i))  # No repeated elements
            SRMSE = 0 
            for tuple in tuples:
                SRMSE += srmse(
                    target.drop(list(columns.difference(tuple)), axis=1),
                    df.drop(list(columns.difference(tuple)), axis=1)
                )
            SRMSE /= len(tuples)
            srmse_dict[model]["SRMSE "+str(i)] = SRMSE
        srmse_dict[model]["Zeros"] = sampling_zeros(source, target, df)
    
    table = []
    for model in srmse_dict:
        table.append(
            pd.DataFrame({i:srmse_dict[model][i] for i in srmse_dict[model]}, index=[model]))
    table = pd.concat(table)

    if save:
        table.to_csv(f"{path}/result_table.csv")
        
    return table