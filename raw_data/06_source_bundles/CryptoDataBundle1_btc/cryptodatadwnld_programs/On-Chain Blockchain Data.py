# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 16:24:41 2025

@author: loren
"""


# MIT License
# No express warranty
import pandas as pd
import s3fs
import datetime

bucket_name = 'aws-public-blockchain'


def generate_date_strings(start_date_str):
    """This function will take a starting date string ('2020-01-01') and then create all dates between that initial date
    to the current date and return the dates in a list """
    # Convert the start date string to a datetime object
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.datetime.now()
    # end_date = get_earliest_date(sym=sym)
    if isinstance(end_date, datetime.date):
        end_date = datetime.datetime.combine(end_date, datetime.time())

    # List to hold all date strings
    date_strings = []

    # Generate dates
    while start_date < end_date:
        date_strings.append(start_date.strftime('%Y-%m-%d'))
        start_date += datetime.timedelta(days=1)  # Increment day by one

    return date_strings


def main(chain, start_date):
    """ this is our main function. we are going to create a list of dates to find data for, then pass those in a for loop
    to the get_block_data function to get the actual data for that date.
    """
    mydata = []
    date_list = generate_date_strings(start_date_str=start_date)

    for d in date_list:
        mydata.append(get_block_data(chain=chain, dt=d))


def get_block_data(chain, dt):
    """This gets block data for a particular date and chain (either 'btc' or 'eth')
    using the public amazon web buckets

    """
    # we are going to extract from the aws s3 web bucket
    file_key = f'v1.0/{chain}/blocks/date={dt}/'
    s3 = s3fs.S3FileSystem(anon=True) # create an s3 object
    base_path = f's3://{bucket_name}/{file_key}'  # create the path
    files = [f for f in s3.ls(base_path) if f.endswith('.parquet')]  # get the parquet filenames in a list

    # Initialize an empty DataFrame
    df = pd.DataFrame()

    # Read each Parquet file into the DataFrame
    print(f'Extracting {chain} block data for {dt}')

    for file in files:  # for all of thefiles, lets loop over and read into pandas dataframe
        file_path = f's3://{file}'
        df = pd.concat([df, pd.read_parquet(file_path, engine='pyarrow', filesystem=s3)])

    filename = f'Blockchain_{chain}_blocks_{dt}.csv'
    df.to_csv(filename, index=False)
    print(f'Saved {filename}')


if __name__ == "__main__":
    STARTING_DATE = '2024-04-02'  # date to pull starting from
    main(chain='btc', start_date=STARTING_DATE)