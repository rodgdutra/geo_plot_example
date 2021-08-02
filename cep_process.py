import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pycep_correios
from pycep_correios.exceptions import BaseException
from geopy.geocoders import Nominatim, ArcGIS, GoogleV3
import time


def extract_cep_data(address, tries=0, city_only=False):
    geolocator = Nominatim(user_agent="application")
    location = geolocator.geocode(address, language='pt')
    if tries > 1:
        return False, False, False, False

    # If is none tries again
    if location is None:
        address = ','.join(address.split(',')[1:])
        state, region, country, location = extract_cep_data(address,
                                                            tries=tries + 1,
                                                            city_only=True)
        return state, region, country, location

    else:
        try:
            if not city_only:
                state = location.address.split(',')[-4]
                region = location.address.split(',')[-3]
                country = location.address.split(',')[-1]
            else:
                state = location.address.split(',')[-3]
                region = location.address.split(',')[-2]
                country = location.address.split(',')[-1]

            if pycep_correios.validate_cep(region):
                state = location.address.split(',')[-4]
                region = location.address.split(',')[-3]

            if region.split(' ')[1] != 'Região' and country.split(
                    ' ')[1] == 'Brasil':
                state = location.address.split(',')[-3]
                region = location.address.split(',')[-2]

        except IndexError:
            state = location.address.split(',')[0]
            region = location.address.split(',')[1]
            country = location.address.split(',')[2]

        return state, region, country, location


def cep_to_data(df):

    for index, row in df.iterrows():
        if row['state'] == '0':
            cep = row["cep"]
            if pycep_correios.validate_cep(cep):
                try:
                    address = pycep_correios.get_address_from_cep(cep)
                    address = f"{address['logradouro']}, {address['cidade']}, {address['uf']}"
                except BaseException:
                    address = cep
            else:
                address = cep
            # check if is cep or streat and number
            state, region, country, location = extract_cep_data(address)
            if state:
                df['state'][index] = state
                df['country'][index] = country
                df['region'][index] = region
                df['lat'][index] = location.latitude
                df['lon'][index] = location.longitude
                time.sleep(0.05)
                print(state, region, country)
                print(len(location.address.split(',')))
            else:
                print(
                    f'Index :{index}, cep {cep}, {pycep_correios.validate_cep(cep)}'
                )
    return df


def main():
    df = pd.read_csv('new_ceps_3.csv')
    df = cep_to_data(df)
    print(df.head())
    df.to_csv('new_ceps_4.csv')


if __name__ == '__main__':
    main()