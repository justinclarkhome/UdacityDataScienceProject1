import pandas as pd
import numpy as np
import re
import json
from datetime import datetime as dt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import plotly.io as pio

day_map = {0: 'Mon', 1: 'Tues', 2: 'Wed', 3: 'Thurs', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

def add_seasonality_features(df, date_col='date'):
    """ Add categorical seasonality info.

    Loop over the date field and add additional columns reflecting the weekday, month and year of each observation.
    Return the modified dataframe with the new fields. If the date_col label is not present in df, no processing will occur
    and the df will be returned unchanged.
    """
    if date_col not in df:
        print(f'date_col label ({date_col}) not found in df! Not processing seasonality info.')
    else:
        df[date_col] = [pd.Timestamp(i) if type(i) is str else i for i in df[date_col]]
        if 'weekday' not in df:
            df['weekday'] = [i.weekday() for i in df[date_col]]
        if 'month' not in df:
            df['month'] = [i.month for i in df[date_col]]
        if 'year' not in df:
            df['year'] = [i.year for i in df[date_col]]
    return df


def generate_choropleth(
    df,
    geojson_file='./data/airbnb_boston/boston.geojson',
    output_filename='prices_and_neighborhoods.png',
    use_log_price=True,
):
    """ Generates chorpleth plot of neighborhoods with listing details overlaid.

    Thank you for the geojson data: https://github.com/codeforgermany/click_that_hood/blob/main/public/data/boston.geojson?short_path=46589b4
    If output_filename is included, a static version of the plot will be saved to that filename.
    """
    with open(geojson_file, 'r') as f:
        geojson = json.load(f)
        
    geoplot_data = df[['latitude', 'longitude', 'zipcode_cleaned', 'price']].dropna()
    geoplot_data['log_price'] = geoplot_data['price'].apply(np.log)
    
    fig = px.choropleth(
        data_frame={'name': [i['properties']['name'] for i in geojson['features']]}, 
        geojson=geojson, 
        locations='name', 
        featureidkey="properties.name",
        title='Boston Neighborhoods and Airbnb Prices<br>Colorbar is Log Scale',
    )
    fig.update_geos(fitbounds="locations", visible=False) 
    fig.add_trace(
        px.scatter_geo(
            data_frame=geoplot_data, 
            lat='latitude', 
            lon='longitude', 
            color='log_price',
            hover_data={
                'latitude': ':.2f', 
                'longitude': ':.2f', 
                'price': ':.2f', 
                'log_price': ':.2f',
            },
        ).data[0])
    
    # relabel the colorbar (as showing log values is confusing)
    fig.update_coloraxes(colorbar={
        'title': 'Price',
        'tickvals': geoplot_data['log_price'].quantile([0.01, 0.999]).values,
        'ticktext': ['Cheaper', 'Pricier'],
    })
    fig.update_layout(showlegend=False)
    
    fig.show()

    if output_filename is not None:
        print(f'Writing {output_filename}')
        pio.write_image(fig, output_filename) 


def load_and_process_raw_data():
    """ Load underlying Airbnb Boston data and apply some processing steps.
    
    The function will attempt to convert string values that look like prices (e.g. "$5,000") to a float, where possible.
    Weekly and monthly prices, where missing, will be estiamted via OLS regression based on the price field.
    Missing zipcodes will be estimated via KNN classified using latitude and longitude values.

    The function will returb three cleaned dataframes: listings, calendar and reviews.
    """
    listings = pd.read_csv('./data/airbnb_boston/listings.csv', index_col='id')
    calendar = pd.read_csv('./data/airbnb_boston/calendar.csv')
    reviews = pd.read_csv('./data/airbnb_boston/reviews.csv')

    listings = listings.map(
        convert_dollars_to_float).map(convert_percentages_to_float).map(convert_string_date_to_dt)
    calendar = calendar.apply(
        convert_dollars_to_float).map(convert_percentages_to_float).map(convert_string_date_to_dt)
    reviews = reviews.apply(
        convert_dollars_to_float).map(convert_percentages_to_float).map(convert_string_date_to_dt)
    
    # estimate missing monthly_price/weekly_price fields using regression (based on price field)
    listings = estimate_y_from_X_ols(
        data=listings, y_label='monthly_price', X_labels='price')['filled_data']
    listings = estimate_y_from_X_ols(
        data=listings, y_label='weekly_price', X_labels='price')['filled_data']
    
    # estimate missing zip codes using KNN classification (based on latitude and longitude)
    listings = classify_y_based_on_X_knn(
        data=get_cleaned_zipcodes(listings), 
        y_label='zipcode_cleaned', X_labels=['latitude', 'longitude'],
    )['filled_data']    

    # convert price column to a float, assuming format of $#.#
    calendar.price = [float(i.replace('$', '').replace(',', '')) if type(i) is str else i for i in calendar.price]

    # add a month, year columns for seasonality analysis
    calendar = add_seasonality_features(df=calendar)

    return listings, calendar, reviews
    

def get_columns_and_types(df):
    """
    Loop over the dtypes of the columns of a dataframe, and return a dictionary of lists identifying which are ints, floats and objects.
    """
    print(f'Types detected: {", ".join([str(i) for i in df.dtypes.unique()])}')
    return {
        'int': [label for label, dtype in df.dtypes.items() if dtype in [int, np.int64]],
        'float': [label for label, dtype in df.dtypes.items() if dtype in [float, np.float64]],
        'object': [label for label, dtype in df.dtypes.items() if dtype in ['O', 'object']],
    }


def convert_dollars_to_float(s, pattern=r"\$|,"):
    """ Convert money-like strings to floats."""
    if type(s) is str and re.match(pattern, s):
        # if the input is a str containing '$' and/or ',' try to remove those chars and covert the result to a float
        # if this fails, then there is likely text mixed in (like "$195 this week only!")
        try:
            return float(re.sub(pattern, repl='', string=s))
        except:
            return s
    else:
        # otherwise just return the input
        return s


def estimate_y_from_X_ols(data, y_label, X_labels, train_size=0.6, random_state=42, add_constant=False):
    """
    Run a linear regression on data using y_label as the dependent variable and X_labels as the independent variable(s): y_label ~ X_labels
    A constant can be optionally added, but note that the r2 test may give misleading results when a constant is included.
    """
    if type(X_labels) not in [list, tuple]:
        X_labels = [X_labels] # in case we pass a scalar
        
    # we know that we have data for 'price' in all observations, but not for 'monthly_price' or 'weekly_price'
    # drop the nans, create a train/test split, build a model and estimate the monthly, then use the model to fill the missing values in the dataset
    reg_data = data[X_labels + [y_label]].dropna()
    y = reg_data[y_label]
    X = reg_data[X_labels]
    if add_constant:
        X = sm.add_constant(reg_data[X_labels])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)

    print(f'Estimating {y_label}~1+{"+".join(X_labels)}')
    m = sm.OLS(exog=X_train, endog=y_train).fit()
    
    print(f'... Train fit: {r2_score(y_true=y_train, y_pred=m.predict(exog=X_train)):.2f}')
    print(f'... Test fit: {r2_score(y_true=y_test, y_pred=m.predict(exog=X_test)):.2f}')
    
    print(f'... Filling NaNs in {y_label} with estimated data')
    missing = data[y_label][data[y_label].isnull()].index
    if add_constant:
        data.loc[missing, y_label] = m.predict(exog=sm.add_constant(data[data[y_label].isnull()][X_labels]))
    else:
        data.loc[missing, y_label] = m.predict(exog=data[data[y_label].isnull()][X_labels])
    
    return {
        'filled_data': data,
        'model': m,
        'X_train': X_train, 
        'X_test': X_test, 
        'y_train': y_train,
        'y_test': y_test,
    }


def convert_percentages_to_float(s):
    """
    Take in a value, and if it is a string ending with a percent side, strip the percent sign and return a float. 
    If the conversion fails, return the original value.
    """
    if type(s) is str and s.endswith('%'):
        try:
            return float(s.strip('%'))
        except:
            return s
    else:
        return s


def convert_string_date_to_dt(s):
    """
    Take in a value, and if it is a string that looks like a date in YYYY-MM-DD format, convert it to a datetime object.
    If the conversion fails, return the original value.
    """
    try:
        return dt.strptime(s, '%Y-%m-%d')
    except:
        return s

        
def get_cleaned_zipcodes(data):
    """
    Scan over the values of 'zipcodes' in the data, and isolate those that are 5 characters long, with other values
    return as NaN. This dataset is pretty clean with zips, but there's at least 1 entry where the zip is more than 10 digits.
    Store the cleaned values as 'zipcode_cleaned'. Note that if the 'zipcode_cleaned' field is aready in data, it will be dropped 
    and recalculated.

    The 'zipcode_cleaned' field - including the nans - will be used later to feed into a classifier (to get estimates for the nans 
    based on latitude and longitude).
    """
    if 'zipcode_cleaned' in data:
        data = data.drop('zipcode_cleaned', axis=1)

    new_zipcodes = []
    for k, v in data['zipcode'].items():
        if pd.isnull(v) or type(v) is str and len(v) != 5:
            new_zipcodes.append(np.nan)
        else:
            new_zipcodes.append(str(v))
    data = pd.concat([
        data,
        pd.Series(index=data.index, data=new_zipcodes).to_frame('zipcode_cleaned').astype('Int64'),
    ], axis=1)
    return data
    

def classify_y_based_on_X_knn(data, y_label, X_labels, train_size=0.6, random_state=42):
    """
    Wrapped for a K nearest neighbor classifier, that will estimate categories contained in y_label column of data, 
    using the variables contained in X labels columns of data.
    """

    if type(X_labels) is str:
        X_labels = [X_labels]
    
    knn_data = data[[y_label] + X_labels].dropna()
    y = knn_data[y_label]
    X = knn_data[X_labels]    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)
    
    print(f'Estimating {y_label} labels from {"+".join(X_labels)}')
    m_knn = KNeighborsClassifier()
    m_knn.fit(X_train, y_train)
    
    print(f'... Train fit: {m_knn.score(X_train, y_train):.2f}')
    print(f'... Test fit: {m_knn.score(X_test, y_test):.2f}')
    
    print(f'... Filling NaNs in {y_label} with estimated data')
    missing = data[y_label][data[y_label].isnull()].index
    data.loc[missing, y_label] = m_knn.predict(data[data[y_label].isnull()][X_labels])
    
    return {
        'filled_data': data,
        'model': m_knn,
        'X_train': X_train, 
        'X_test': X_test, 
        'y_train': y_train,
        'y_test': y_test,
    }
