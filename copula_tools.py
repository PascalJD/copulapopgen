import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF


class CopulaScaler():
    """
    This class can encode and decode features using the probability integral transform
    and the pseudo-inverse of the provided data. It can also resample discrete encoded
    features uniformly between the steps of their Empirical Cumulative Distribution Function (ECDF).
    """
        
    def fit(self, data):
        """
        Compute the ECDFs and sorted pairs (x, ECDF(x)) used for scaling.

        Parameters
        ----------
        data : pd.DataFrame
            The data for which to compute ECDFs and pseudo-inverses.

        Returns
        -------
        self : object
            Fitted scaler object.
        """
        self.data = data
        self.columns = data.columns
        self.ecdfs = {}  # Empirical CDF as a step function.
        self.ecdf_map = {}  # {column : [(x, ecdf(x)), ...], ...} sorted

        for column in self.columns:
            feature = data[column]
            # ECDF
            ecdf = ECDF(feature)
            self.ecdfs[column] = ecdf
            # Create the mapping for the pseudo inverse
            x_u = []  # [(x_1, ECDF(x_1)), ..., (x_d, ECDF(x_d))] with x_i < x_j for i < j
            uniques = np.sort(feature.unique())  # [x_1, ..., x_d]
            for x in uniques:
                u = ecdf(x)
                x_u.append((x, u))
            self.ecdf_map[column] = x_u

        return self
    
            
    def transform(self, data):
        """Apply the probability integral transform on the marginal distributions.

        Parameters
        ----------
        data : pd.DataFrame
            The data used to scale,

        Returns
        -------
        data_tr : pd.DataFrame
            The transformed data 
        """
        data_tr = data.copy()
        for column in self.columns:
            data_tr[column] = data_tr[column].apply(self.ecdfs[column])
        return data_tr

    
    def inverse_transform(self, data):
        """Apply the pseudo-inverse on the marginal distributions.

        Parameters
        ----------
        data : pd.DataFrame 
            The scaled data to be mapped back to the original space.

        Returns
        -------
        data_tr : pd.DataFrame
            The trasnformed data
        """
        data_tr = data.copy()
        for column in self.columns:
            x_u = np.array(self.ecdf_map[column])
            data_tr[column] = data_tr[column].apply(CopulaScaler.pseudo_inverse, args=(x_u,))
        return data_tr


    def interpolation(self, data, categorical):
        """Linear interpolations between consecutive steps of the ECDF.

        Parameters
        ----------
        data : pd.DataFrame
            The data for which to interpolate

        categorical : array-like
            The discrete features to interpolate

        Returns
        -------
        data_tr : pd.DataFrame
            The data with interpolated discrete features 
        """

        def sample_uniform(u, x_u):
            us = x_u[:, 1]
            idx = list(us).index(u)
            if idx == 0:
                return np.random.uniform(low=0, high=u)
            else:
                return np.random.uniform(low=us[idx-1], high=u)

        data_tr = data.copy()
        for column in categorical:
            x_u = np.array(self.ecdf_map[column])
            data_tr[column] = data_tr[column].apply(sample_uniform, args=(x_u,))
        return data_tr


    @staticmethod
    def pseudo_inverse(u, x_u):
        """Compute pseudo inverse.

        Parameters
        ----------
        u : float
            The value for which to compute the pseudo-inverse

        x_u : array-like of Tuples
            Array of pairs (x, ECDF(x)) 

        Returns
        -------
        x : float
            The pseudo-inverse value of u given the ECDF x_u
        """
        us = x_u[:, 1]
        valid_i = np.ma.MaskedArray(us, us<u)
        x = x_u[np.ma.argmin(valid_i), 0]
        return x
