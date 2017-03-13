import pandas as pd

valid_years = ['09', '10', '11', '12', '13', '14']


def prepare_data(soy, geo, gene, years=valid_years, previous=1):
    # create the joint data_frame that we will use to fit the model
    pd_soy = soy[['YEAR', 'LOCATION', 'VARIETY', 'RM', 'YIELD']]
    pd_geo = geo.drop(['LATITUDE', 'LONGITUDE', 'FIPS', 'AREA'], 1)
    pd_gene = gene.drop(['RM', 'CLASS_OF'], 1)

    # convert gene information into integer values
    pd_gene = pd_gene.replace(to_replace=['AA', 'CC', 'GG', 'TT', 'AC', 'AG', 'AT', 'CG', 'NN'],
                              value=[1, 2, 3, 4, 5, 6, 7, 8, 0])

    # merge tables
    pd_soy_geo = pd.merge(pd_soy, pd_geo, on='LOCATION')
    # TODO now we use inner join here because some VARIETY are not present in the
    # gene table. Some similarity estimation like KNN shall be used in the future
    pd_joint = pd.merge(pd_soy_geo, pd_gene, how='inner', on='VARIETY')

    # training data
    training = pd_joint.drop(['LOCATION', 'VARIETY'], 1)

    def training_data_for_year(_years, _previous=1):
        """
        Select data from training set that come from given years.
        All geo info in year X will be dropped except those of year X
        and some years which are previous to year X.

        For example:
        If the original training data is like

        =============================================
        | YEAR | TEMP_09 | TEMP_10 | RAD_09 | RAD_10|
        ---------------------------------------------
        | 2010 | 123.123 | 456.456 | 11.234 | 22.111|
        =============================================

        The result should looks like:

        ==========================================
        | YEAR | TEMP-0 | TEMP-1 | RAD-0 | RAD-1 |
        ------------------------------------------
        | 2010 | 456.456| 123.123| 22.111| 11.234|
        ==========================================

        Where TEMP-0 is the temp of current year (2010), and TEMP-1 is
        the temp of current year minus 1 (2009).

        :param _years: Which years you want (e.g. ['09', '10', '11'])
        :param _previous: Leave how many years previous to the corresponding one
        :return: selected data set
        """
        global col
        if not set(_years).issubset(valid_years):
            raise Exception('Invalid year to build training data: '.join(_years))

        # select rows for that year
        int_years = map(lambda year: 2000 + int(year), _years)
        all_years_training = training[training['YEAR'].isin(int_years)]
        result = all_years_training.copy()

        # select geo related columns of the year which is years_ahead of each row's 'YEAR' column
        def info_of_year(col, years_ahead=0):
            col_prefix = col + '_'

            def actual_year(row):
                return str(row['YEAR'] - years_ahead)[2:]

            return lambda row: row[col_prefix + actual_year(row)]

        for i in range(_previous + 1):
            result['TEMP-%d' % i] = all_years_training.apply(info_of_year('TEMP', years_ahead=i), axis=1)
            result['PREC-%d' % i] = all_years_training.apply(info_of_year('PREC', years_ahead=i), axis=1)
            result['RAD-%d' % i] = all_years_training.apply(info_of_year('RAD', years_ahead=i), axis=1)

        # drop original 'TEMP_' columns
        col_to_drop = [col for col in list(result.columns.values) if
                       ('TEMP_' in col or 'PREC_' in col or 'RAD_' in col)]
        return result.drop(col_to_drop, axis=1)

    return training_data_for_year(years, previous)


if __name__ == '__main__':
    import sys

    soy_file = pd.read_csv(sys.argv[1])
    geo_file = pd.read_csv(sys.argv[2])
    gene_file = pd.read_csv(sys.argv[3])

    prepare_data(soy_file, geo_file, gene_file).to_csv('training.csv')
