import pandas as pd

valid_years = ['09', '10', '11', '12', '13', '14']


def prepare_data(soy, geo, gene, years=valid_years, previous=1):
    return prepare_geo_data(soy, geo, years, previous)


def prepare_geo_data(soy, geo, years, previous):
    pd_soy = soy[['YEAR', 'LOCATION', 'VARIETY', 'RM', 'YIELD']]
    pd_geo = geo.drop(['LATITUDE', 'LONGITUDE', 'FIPS', 'AREA'], 1)

    # merge soy and geo tables
    pd_soy_geo = pd.merge(pd_soy, pd_geo, on='LOCATION')
    training = pd_soy_geo.drop(['LOCATION'], 1)

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
        if not set(_years).issubset(valid_years):
            raise Exception('Invalid year to build training data: '.join(_years))

        # select rows of given years
        int_years = map(lambda year: 2000 + int(year), _years)
        all_years_training = training[training['YEAR'].isin(int_years)]
        result = all_years_training.copy()

        # select geo related columns of the year which is years_ahead of each row's 'YEAR' column
        def info_of_year(column, years_ahead=0):
            col_prefix = column + '_'

            def actual_year(row):
                return str(row['YEAR'] - years_ahead)[2:]

            return lambda row: row[col_prefix + actual_year(row)]

        for i in range(_previous + 1):
            result['TEMP-%d' % i] = all_years_training.apply(info_of_year('TEMP', years_ahead=i), axis=1)
            result['PREC-%d' % i] = all_years_training.apply(info_of_year('PREC', years_ahead=i), axis=1)
            result['RAD-%d' % i] = all_years_training.apply(info_of_year('RAD', years_ahead=i), axis=1)

        # drop original 'TEMP_'-like columns
        col_to_drop = [col for col in list(result.columns.values) if
                       ('TEMP_' in col or 'PREC_' in col or 'RAD_' in col)]
        return result.drop(col_to_drop, axis=1)

    return training_data_for_year(years, previous)


def most_significant_gene(gene, num):
    def cal_sd(s):
        """
        Calculate the 'unlikeability' of each gene column.
        Here we use :
        sd = 1 - p1^2 - p2^2 ... - pm^2,
        where pm means the frequency of category m.
        :param s:
        :return:
        """
        n = s.sum()
        d = 0
        for i, c in s.iteritems():
            freq = 1.0 * c / n
            d += freq * freq
        return 1 - d

    pd_gene = gene.drop(['VARIETY', 'FAMILY', 'RM', 'CLASS_OF'], 1)
    l = []

    for col in pd_gene:
        entry = (col, cal_sd(pd_gene[col].value_counts()))
        l.append(entry)

    return [e[0] for e in sorted(l, key=lambda tup: tup[1], reverse=True)[:num]]


if __name__ == '__main__':
    import sys

    soy_file = pd.read_csv('dataset/yield.csv')
    geo_file = pd.read_csv('dataset/geo.csv')
    gene_file = pd.read_csv('dataset/gene.csv')

    if len(sys.argv) == 1:
        print 'Usage: python prepare.py [geo]'
        exit(1)

    if sys.argv[1] == 'geo':
        output_file = 'training.csv'
        prepare_data(soy_file, geo_file, gene_file).to_csv(output_file)
        print 'Prepare geo data done. output to file "%s"' % output_file
    else:
        print 'Wrong command'
        exit(1)
