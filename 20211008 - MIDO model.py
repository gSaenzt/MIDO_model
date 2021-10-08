##############################################################################################
#                                                                                            #
#                           #ALL PACKAGES NECESSARY FOR A RUN                                #
#                                                                                            #
##############################################################################################

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import datetime
import pandas as pd
import numpy as np
import numpy_financial as npf
from time import sleep
from shutil import copy
import os
import io
from matplotlib import pyplot as plt
import copy as cpy
from scipy.cluster.hierarchy import dendrogram, linkage
# from numpy import *
from shapely.geometry import Point, LineString
from sklearn.metrics import pairwise_distances_argmin_min
from string import digits
import glob
import pathlib
import subprocess
from datetime import datetime

##############################################################################################
#                                                                                            #
#                   #ALL INPUT VALUES THAT CAN CHANGE - PART OF FUNCTION INPUT               #
#                                                                                            #
##############################################################################################

year = 1
end_year = 20
experiment = 203

standard_capacity_dic = {'OffshoreWind': 600, 'OnshoreWind': 600, 'SolarPV': 200, 'GasCCGT': 800, 'GasCCGTCCS': 800,
                         'GasOCGT': 300, 'Biomass': 500, 'Nuclear': 1000, 'LiOnStorage': 258, 'ElectrolyserS': 200,
                         'ElectrolyserCCGT': 450, 'ElectrolyserOCGT': 200, 'LiOnStorageCharge': 200,
                         'LionStorageDischarge': 200, 'ElectroyerSCharge': 200, 'ElectrolyserSDischarge': 200,
                         'ElectrolyserCCGTCharge': 450, 'ElectrolyserCCGTDischarge': 450, 'ElectrolyserOCGTCharge': 200,
                         'ElectrolyserOCGTDischarge': 200}

CO2_dic = {'OffshoreWind': 0, 'OnshoreWind': 0, 'SolarPV': 0, 'GasCCGT': 0.2016, 'GasCCGTCCS': 0, 'GasOCGT': 0.2016,
           'Biomass': 0, 'Nuclear': 0, 'LiOnStorage': 0, 'ElectrolyserS': 0, 'ElectrolyserCCGT': 0,
           'ElectrolyserOCGT': 0}

euro_mw_dic = {'OffshoreWind': 2.102, 'OnshoreWind': 1.137, 'SolarPV': 0.395, 'GasCCGT': 0.9, 'GasCCGTCCS': 1.7,
               'GasOCGT': 0.4,
               'Biomass': 2, 'Nuclear': 7, 'LiOnStorage': 0.2495, 'ElectrolyserS': 0.662, 'ElectrolyserCCGT': 1.607,
               'ElectrolyserOCGT': 1.082}

fix_OM_per_dic = {'OffshoreWind': 0.03, 'OnshoreWind': 0.03, 'SolarPV': 0.01, 'GasCCGT': 0.03, 'GasCCGTCCS': 0.03,
                  'GasOCGT': 0.03,
                  'Biomass': 0.04, 'Nuclear': 0.03, 'LiOnStorage': 0.01, 'ElectrolyserS': 0.01,
                  'ElectrolyserCCGT': 0.03,
                  'ElectrolyserOCGT': 0.03}

asset_lifetime_dic = {'OffshoreWind': 25, 'OnshoreWind': 25, 'SolarPV': 25, 'GasCCGT': 45, 'GasCCGTCCS': 45,
                      'GasOCGT': 45, 'Biomass': 40, 'Nuclear': 60, 'LiOnStorage': 15, 'ElectrolyserS': 20,
                      'ElectrolyserCCGT': 30, 'ElectrolyserOCGT': 30}

asset_eco_lifetime_dic = {'OffshoreWind': 15, 'OnshoreWind': 15, 'SolarPV': 15, 'GasCCGT': 15, 'GasCCGTCCS': 15,
                          'GasOCGT': 15, 'Biomass': 15, 'Nuclear': 25, 'LiOnStorage': 10, 'ElectrolyserS': 15,
                          'ElectrolyserCCGT': 20, 'ElectrolyserOCGT': 20}

asset_capacity_dic = {'OffshoreWind': 19, 'OnshoreWind': 10, 'SolarPV': 133, 'GasCCGT': 18, 'GasCCGTCCS': 0,
                      'GasOCGT': 26, 'Biomass': 1, 'Nuclear': 0, 'LiOnStorage': 20, 'ElectrolyserS': 10,
                      'ElectrolyserCCGT': 0, 'ElectrolyserOCGT': 1}

asset_delay_dic = {'OffshoreWind': 3, 'OnshoreWind': 2, 'SolarPV': 3, 'GasCCGT': 3,
                   'GasCCGTCCS': 4, 'GasOCGT': 2, 'Biomass': 4, 'Nuclear': 9,
                   'LiOnStorage': 2, 'ElectrolyserS': 2, 'ElectrolyserCCGT': 2,
                   'ElectrolyserOCGT': 2}

gas_asset_efficiency = {'GasCCGT': 0.6, 'GasCCGTCCS': 0.52, 'GasOCGT': 0.41}
biomass_asset_effiency = {'Biomass': 0.42}
nuclear_asset_effiency = {'Nuclear': 0.33}

offshore_development = -0.010748663
onshore_development = -0.008555209
solar_development = -0.007940161
gas_CCGT_development = -0.006684492
electrolyser_development = -0.009063444
storage_development = -0.017773893

fp_lookahead = 7
lookbackperiod = 4

energy_companies = 3
equity_factor = 0.3

capacity_mechanism = 0
um_in = 0.025  # upper margin
lm_in = 0.025  # lower margin
Pc = 60000  # €/MW
r = 0.095

VOLL = 4000
budget_year0 = 4e9
dismantle_loop_stop = 0.93


##############################################################################################
#                                                                                            #
#           #ALL INPUT VALUES THAT CAN NOT CHANGE - NOT PART OF FUNCTION INPUT               #
#                                                                                            #
##############################################################################################

investment_list = ['OffshoreWind', 'OnshoreWind', 'SolarPV', 'GasCCGT', 'GasCCGTCCS', 'GasOCGT', 'Biomass',
                   'Nuclear',
                   'LiOnStorage', 'ElectrolyserS', 'ElectrolyserCCGT', 'ElectrolyserOCGT']

investment_list_classifier_dic = {'OffshoreWind': 'VRES', 'OnshoreWind': 'VRES', 'SolarPV': 'VRES',
                                  'GasCCGT': 'Asset',
                                  'GasCCGTCCS': 'Asset', 'GasOCGT': 'Asset', 'Biomass': 'Asset', 'Nuclear': 'Asset',
                                  'LiOnStorage': 'Storage', 'ElectrolyserS': 'Storage',
                                  'ElectrolyserCCGT': 'Storage',
                                  'ElectrolyserOCGT': 'Storage'}

storage_list_classifier_dic = {'OffshoreWind': 'NotHydrogen', 'OnshoreWind': 'NotHydrogen',
                               'SolarPV': 'NotHydrogen',
                               'GasCCGT': 'NotHydrogen', 'GasCCGTCCS': 'NotHydrogen', 'GasOCGT': 'NotHydrogen',
                               'Biomass': 'NotHydrogen', 'Nuclear': 'NotHydrogen', 'LiOnStorage': 'NotHydrogen',
                               'ElectrolyserS': 'Hydrogen', 'ElectrolyserCCGT': 'Hydrogen',
                               'ElectrolyserOCGT': 'Hydrogen'}

investment_storage_dic = {'LiOnStorage': ['LiOnStorageCharge', 'LionStorageDischarge'],
                          'ElectrolyserS': ['ElectroyerSCharge', 'ElectrolyserSDischarge'],
                          'ElectrolyserCCGT': ['ElectrolyserCCGTCharge', 'ElectrolyserCCGTDischarge'],
                          'ElectrolyserOCGT': ['ElectrolyserOCGTCharge', 'ElectrolyserOCGTDischarge']}

charge_storage_dic = {'LiOnStorageCharge': 'Charge', 'ElectroyerSCharge': 'Charge',
                      'ElectrolyserCCGTCharge': 'Charge',
                      'ElectrolyserOCGTCharge': 'Charge'}

discharge_storage_dic = {'LionStorageDischarge': 'Discharge', 'ElectrolyserSDischarge': 'Discharge',
                         'ElectrolyserCCGTDischarge': 'Discharge', 'ElectrolyserOCGTDischarge': 'Discharge'}

fuel_search_dic = {'OffshoreWind': 'Free', 'OnshoreWind': 'Free', 'SolarPV': 'Free', 'GasCCGT': 'Gas',
                   'GasCCGTCCS': 'Gas', 'GasOCGT': 'Gas', 'Biomass': 'Biomass', 'Nuclear': 'Uranium',
                   'LiOnStorage': 'Free', 'ElectrolyserS': 'Free', 'ElectrolyserCCGT': 'Free',
                   'ElectrolyserOCGT': 'Free'}

asset_linny_output_dic = {'OffshoreWind': 'windoffshoreparkL', 'OnshoreWind': 'windonshoreparkL',
                          'SolarPV': 'solarparkL', 'GasCCGT': 'GasCCGTL', 'GasCCGTCCS': 'GasCCGTCCSL',
                          'GasOCGT': 'GasOCGTL', 'Biomass': 'BiomassL', 'Nuclear': 'NuclearL',
                          'LiOnStorageCharge': 'LiOnchargeL', 'LionStorageDischarge': 'LiOndischargeL',
                          'ElectroyerSCharge': 'ElectrolyserSL',
                          'ElectrolyserSDischarge': 'transportfromstoragetonetworkL',
                          'ElectrolyserCCGTCharge': 'ElectrolyserCCGTL',
                          'ElectrolyserCCGTDischarge': 'HydrogenCCGTL',
                          'ElectrolyserOCGTCharge': 'ElectrolyserOCGTL',
                          'ElectrolyserOCGTDischarge': 'HydrogenOCGTL'}

asset_mc_linny_output_dic = {'OffshoreWind': 'windoffshoreparkL', 'OnshoreWind': 'windonshoreparkL',
                             'SolarPV': 'solarparkL', 'GasCCGT': 'GasCCGTL', 'GasCCGTCCS': 'GasCCGTCCSL',
                             'GasOCGT': 'GasOCGTL', 'Biomass': 'BiomassL', 'Nuclear': 'NuclearL'}

weather_dic = {'OffshoreWind': 'Offshore', 'OnshoreWind': 'Onshore', 'SolarPV': 'Solar'}

government_asset_list = []

# Determine ages of all assets
solar_age_list = [20, 18, 17, 17, 16, 16, 15, 15, 15, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 11,
                  11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8,
                  8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5,
                  5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
wind_onshore_age_list = [24, 24, 21, 16, 14, 9, 6, 4, 3, 2, 1]
wind_offshore_age_list = [23, 22, 15, 14, 10, 10, 9, 8, 7, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
CCGT_age_list = [45, 40, 37, 35, 33, 31, 26, 24, 20, 20, 18, 17, 17, 8, 6, 5, 2, 2]

OCGT_age_list = [42, 40, 37, 36, 35, 34, 33, 33, 32, 31, 31, 26, 26, 23, 23, 20, 20, 20, 18, 18, 18, 17, 17, 8, 5, 2]
ElectrolyserS_age_list = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
LiOnS_age_list = [10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]


CO2_price = 35
CO2_development = 1.062
gas_price = 25.56
gas_development = 1.011
uranium_price = 12.96
uranium_development = 1.035
biomass_price = 21.6
biomass_development = 1.062
hydrogen_import_price = 4000
hydrogen_development = 1

FP_CO2_price = cpy.deepcopy(CO2_price)
FP_biomass_price = cpy.deepcopy(biomass_price)
FP_hydrogen_import_price = cpy.deepcopy(hydrogen_import_price)
FP_gas_price = cpy.deepcopy(gas_price)
FP_uranium_price = cpy.deepcopy(uranium_price)

electricity_demand_change = 1.0225
hydrogen_demand_change = 1.138

asset_only_MC_dic = {'Free': 0, 'Biomass': biomass_price, 'Gas': gas_price, 'Uranium': uranium_price,
                     'Hydrogen_import': hydrogen_import_price}

delay_dic = {}
budget_number = 0

df_dismantle = pd.DataFrame()

df_government_solar = pd.read_csv(r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\government\solar_gov.csv")
df_goverment_onshore = pd.read_csv(r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\government\onshore_gov.csv")
df_government_offshore = pd.read_csv(r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\government\offshore_gov.csv")

##############################################################################################
#                                                                                            #
#                           #MAKE ALL DICTIONARIES/LIST TO RUN CODE                          #
#                                                                                            #
##############################################################################################

all_asset_capacity_dic = {}
asset_MC_dic = {}
skip = 0
receiver_path = [r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver"]
original_path = pathlib.Path().resolve()

investment_list_list = []
investment_costs_dic = {}
asset_OM_dic = {}

investment_list_dic = {}

for assets in investment_list:
    investment_costs_dic[assets] = standard_capacity_dic[assets] * euro_mw_dic[assets] * 1e6

key_min = min(investment_costs_dic.keys(), key=(lambda k: investment_costs_dic[k]))
min_cost = investment_costs_dic[key_min]

for assets in investment_list:
    asset_OM_dic[assets] = investment_costs_dic[assets] * fix_OM_per_dic[assets]

for assets in investment_list:
    investment_list_list += [str(assets) + '_asset_list']

investment_list_dic = {key: [] for key in investment_list_list}

for assets in investment_list_dic:
    for asset in investment_list:
        remove = len(assets) - len('_asset_list')
        if asset == assets[:remove]:
            for i in reversed(range(asset_capacity_dic[asset])):
                investment_list_dic[assets] += ([str(asset) + str(i + 1)])

for asset_groups in investment_list_dic:
    for individual_assets in investment_list_dic[asset_groups]:
        all_asset_capacity_dic[individual_assets] = 1

installed_counter_list = []
installed_counter_dic = {}
all_asset_age_dic = {}

for asset_groups in investment_list_dic:
    for individual_assets in investment_list_dic[asset_groups]:
        all_asset_age_dic[individual_assets] = 1

all_asset_age_dic['Biomass1'] = 20
all_asset_age_dic['ElectrolyserOCGT1'] = 5

for assets in investment_list:
    installed_counter_list += [str(assets) + '_installed_counter']

installed_counter_dic = {key: [] for key in installed_counter_list}

for asset_groups_counter in installed_counter_dic:
    installed_counter_dic[asset_groups_counter] = 0

for asset_groups_counter in installed_counter_dic:
    for asset_groups in investment_list_dic:
        remove1 = len(asset_groups_counter) - len('_installed_counter')
        remove2 = len(asset_groups) - len('_asset_list')
        if asset_groups_counter[:remove1] == asset_groups[:remove2]:
            counter = 0
            for individual_assets in investment_list_dic[asset_groups]:
                counter += 1
            installed_counter_dic[asset_groups_counter] = counter

investment_list_list = []
investment_list_dic = {}
pipeline_list = []

for assets in investment_list:
    investment_list_list += [str(assets) + '_asset_list']

investment_list_dic = {key: [] for key in investment_list_list}

for assets in investment_list_dic:
    for asset in investment_list:
        remove = len(assets) - len('_asset_list')
        if asset == assets[:remove]:
            for i in reversed(range(asset_capacity_dic[asset])):
                investment_list_dic[assets] += ([str(asset) + str(i + 1)])

for assets in investment_list:
    pipeline_list += [str(assets) + '_in_pipeline']

pipeline_list_dic = {key: 0 for key in pipeline_list}


def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return listOfKeys


def RemoveNumber(s):
    remove_digits = str.maketrans('', '', digits)
    res = s.translate(remove_digits)
    return (res)

for asset_groups in investment_list_dic:
    remove1 = len(asset_groups) - len('_asset_list')
    if asset_groups[:remove1] == 'SolarPV':
        for individual_assets in range(len(investment_list_dic[asset_groups])):
            all_asset_age_dic['SolarPV' + str(individual_assets + 1)] = solar_age_list[individual_assets]
    if asset_groups[:remove1] == 'OnshoreWind':
        for individual_assets in range(len(investment_list_dic[asset_groups])):
            all_asset_age_dic['OnshoreWind' + str(individual_assets + 1)] = wind_onshore_age_list[individual_assets]
    if asset_groups[:remove1] == 'OffshoreWind':
        for individual_assets in range(len(investment_list_dic[asset_groups])):
            all_asset_age_dic['OffshoreWind' + str(individual_assets + 1)] = wind_offshore_age_list[
                individual_assets]
    if asset_groups[:remove1] == 'GasCCGT':
        for individual_assets in range(len(investment_list_dic[asset_groups])):
            all_asset_age_dic['GasCCGT' + str(individual_assets + 1)] = CCGT_age_list[individual_assets]
    if asset_groups[:remove1] == 'GasOCGT':
        for individual_assets in range(len(investment_list_dic[asset_groups])):
            all_asset_age_dic['GasOCGT' + str(individual_assets + 1)] = OCGT_age_list[individual_assets]
    if asset_groups[:remove1] == 'LiOnStorage':
        for individual_assets in range(len(investment_list_dic[asset_groups])):
            all_asset_age_dic['LiOnStorage' + str(individual_assets + 1)] = LiOnS_age_list[individual_assets]
    if asset_groups[:remove1] == 'ElectrolyserS':
        for individual_assets in range(len(investment_list_dic[asset_groups])):
            all_asset_age_dic['ElectrolyserS' + str(individual_assets + 1)] = ElectrolyserS_age_list[
                individual_assets]

df_key_indicators = pd.DataFrame()
df_merit_order = pd.DataFrame()

end_year_list = []

for input_year in range(end_year):
    end_year_list += [input_year + 1]

df_key_indicators = pd.DataFrame(
    columns=['supply_ratio', 'shortage_volume', 'shortage_hours', 'mean_e_price', 'CM_volume', 'CM_price',
             'total_CM_costs', 'total_costs'], index=end_year_list)

df_merit_order = pd.DataFrame(columns=investment_list, index=end_year_list)

files = glob.glob(r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver/*")
for f in files:
    os.remove(f)

df_debt = pd.DataFrame()

for asset_groups in investment_list_dic:
    remove1 = len(asset_groups) - len('_asset_list')
    for individual_assets in investment_list_dic[asset_groups]:
        if all_asset_age_dic[individual_assets] < asset_eco_lifetime_dic[asset_groups[:remove1]]:
            for debt_years in range(year, (
                    (year) + asset_eco_lifetime_dic[asset_groups[:remove1]] - all_asset_age_dic[individual_assets])):
                df_debt.loc[debt_years, individual_assets] = investment_costs_dic[asset_groups[:remove1]] * (
                            1 - equity_factor) / asset_eco_lifetime_dic[asset_groups[:remove1]]

##############################################################################################
#                                                                                            #
#                        #CODE TO RUN PRESENT PRICE PART OF THE MIDO MODEL                   #
#                                                                                            #
##############################################################################################

while year <= end_year:

    # Update investment costs to reflect reference years
    for assets in investment_list:
        if assets == 'OffshoreWind':
            euro_mw_dic[assets] = euro_mw_dic[assets] * ((1 + offshore_development))
        elif assets == 'OnshoreWind':
            euro_mw_dic[assets] = euro_mw_dic[assets] * ((1 + onshore_development))
        elif assets == 'SolarPV':
            euro_mw_dic[assets] = euro_mw_dic[assets] * ((1 + solar_development))
        elif assets == 'GasCCGT':
            euro_mw_dic[assets] = euro_mw_dic[assets] * ((1 + gas_CCGT_development))
        elif assets == 'ElectrolyserS':
            euro_mw_dic[assets] = euro_mw_dic[assets] * ((1 + electrolyser_development))
        elif assets == 'ElectrolyerCCGT':
            euro_mw_dic[assets] = 0.945 + (euro_mw_dic[assets] - 0.945) * ((1 + electrolyser_development))
        elif assets == 'ElectrolyerOCGT':
            euro_mw_dic[assets] = 0.420 + (euro_mw_dic[assets] - 0.420) * ((1 + electrolyser_development))
        elif assets == 'LiOnStorage':
            euro_mw_dic[assets] = euro_mw_dic[assets] * ((1 + storage_development))
        else:
            euro_mw_dic[assets] = euro_mw_dic[assets]

    for assets in investment_list:
        investment_costs_dic[assets] = standard_capacity_dic[assets] * euro_mw_dic[assets] * 1e6

    key_min = min(investment_costs_dic.keys(), key=(lambda k: investment_costs_dic[k]))
    min_cost = investment_costs_dic[key_min]

    for assets in investment_list:
        asset_OM_dic[assets] = investment_costs_dic[assets] * fix_OM_per_dic[assets]

    # Update fuel costs prices
    CO2_price = CO2_price * CO2_development
    gas_price = gas_price * gas_development
    uranium_price = uranium_price * uranium_development
    hydrogen_import_price = hydrogen_import_price * hydrogen_development
    biomass_price = biomass_price * biomass_development

    # Write MC all assets to Linny-R
    for assets in investment_list:
        CO2_costs = (CO2_dic[assets] * CO2_price)
        if investment_list_classifier_dic[assets] != 'Storage':
            if fuel_search_dic[assets] == 'Gas':
                fuel_costs = asset_only_MC_dic[fuel_search_dic[assets]] / gas_asset_efficiency[assets]
            elif fuel_search_dic[assets] == 'Uranium':
                fuel_costs = asset_only_MC_dic[fuel_search_dic[assets]] / nuclear_asset_effiency[assets]
            elif fuel_search_dic[assets] == 'Biomass':
                fuel_costs = asset_only_MC_dic[fuel_search_dic[assets]] / biomass_asset_effiency[assets]
            else:
                fuel_costs = 0

            MC = CO2_costs + fuel_costs

            asset_MC_dic[assets] = MC
        else:
            asset_MC_dic[assets] = 0

    MC_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Marginal costs_output\MC.txt"

    for assets in investment_list:
        with open(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Marginal costs_output\MC.txt",
                'w') as f:
            f.write(str(asset_MC_dic[assets]))

        MC_exp_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Marginal costs_output" + "\\" + str(
            assets) + 'MC.txt'

        copy(MC_dir, MC_exp_dir)

    # Update VOLL and Hydrogen import prices

    with open(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Marginal costs_output\MC.txt",
            'w') as f:
        f.write(str(hydrogen_import_price))

    MC_exp_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Marginal costs_output" + "\\" + 'hydrogen_import' + 'MC.txt'

    copy(MC_dir, MC_exp_dir)

    with open(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Marginal costs_output\MC.txt",
            'w') as f:
        f.write(str(VOLL))

    MC_exp_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Marginal costs_output" + "\\" + 'VOLL' + 'MC.txt'

    copy(MC_dir, MC_exp_dir)

    # Write all capacity to Linny-R for seasonal UC

    capacity_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Capacity_output\_capacity.txt"

    for assets in asset_capacity_dic:
        with open(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Capacity_output\_capacity.txt",
                'w') as f:
            f.write(str(asset_capacity_dic[assets]))

        capacity_exp_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Capacity_output" + "\\" + str(
            assets) + '_capacity.txt'

        copy(capacity_dir, capacity_exp_dir)

    # Insert all standard asset sizes capacities into txt.files in Linny-R *24 for szn UC
    standard_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Standard_asset_output\_standard.txt"

    for assets in asset_capacity_dic:
        with open(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Standard_asset_output\_standard.txt",
                'w') as f:
            f.write(str(standard_capacity_dic[assets] * 24))

        standard_exp_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Standard_asset_output" + "\\" + str(
            assets) + 'standard.txt'
        copy(standard_dir, standard_exp_dir)

    # Write input timeseries to Linny-R
    if year == 1:
        df_onshore = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Onshore windpark output _std_weather_full_year.txt",
            names=['Onshore'])
        df_offshore = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Offshore windpark output_std_weather_full_year.txt",
            names=['Offshore'])
        df_solar = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Solar output one asset_std_weather_full_year.txt",
            names=['Solar'])
        df_edemand = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Electricity demand_input.txt",
            names=['E demand'])
        df_hdemand = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Hydrogen demand_input.txt",
            names=['H demand'])

        copy(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Initial levels\t0_target_HSE_year1.txt",
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Initial levels\t0_target_HSE.txt")

        copy(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Initial levels\t0_target_HSCCGT_year1.txt",
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Initial levels\t0_target_HSCCGT.txt")

        copy(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Initial levels\t0_target_HSOCGT_year1.txt",
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Initial levels\t0_target_HSOCGT.txt")
    else:
        df_onshore = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Onshore windpark output _std_weather_full_year.txt",
            names=['Onshore'])
        df_offshore = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Offshore windpark output_std_weather_full_year.txt",
            names=['Offshore'])
        df_solar = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Solar output one asset_std_weather_full_year.txt",
            names=['Solar'])
        df_edemand = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Electricity demand_full_year.txt",
            names=['E demand'])
        df_hdemand = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Hydrogen demand_full_year.txt",
            names=['H demand'])

    df_UC2050_1 = pd.concat([df_onshore, df_offshore, df_solar, df_edemand, df_hdemand], axis=1)

    df_UC2050_1['Timeseries'] = 1

    df_UC2050_1['Onshore'] = np.where(df_UC2050_1['Timeseries'] > 0,
                                      df_UC2050_1['Onshore'] * (asset_capacity_dic['OnshoreWind']), 0)
    df_UC2050_1['Offshore'] = np.where(df_UC2050_1['Timeseries'] > 0,
                                       df_UC2050_1['Offshore'] * (asset_capacity_dic['OffshoreWind']), 0)
    df_UC2050_1['Solar'] = np.where(df_UC2050_1['Timeseries'] > 0,
                                    df_UC2050_1['Solar'] * (asset_capacity_dic['SolarPV']),
                                    0)
    df_UC2050_1['E demand'] = np.where(df_UC2050_1['Timeseries'] > 0,
                                       df_UC2050_1['E demand'] * (electricity_demand_change),
                                       0)
    df_UC2050_1['H demand'] = np.where(df_UC2050_1['Timeseries'] > 0,
                                       df_UC2050_1['H demand'] * (hydrogen_demand_change), 0)

    # Transform input data with rolling average of week to be input into seasonal UC
    df_UC2050 = df_UC2050_1.append(df_UC2050_1)
    df_rolling_UC = df_UC2050.rolling(168).mean()
    df_rolling_UC.reset_index(drop=True, inplace=True)
    label = range(8760)
    df_smooth_UC = df_rolling_UC.drop(labels=label, axis=0)
    df_smooth_UC.reset_index(drop=True, inplace=True)
    df_smooth_UC.round(decimals=2)

    day_list = []
    year_list = []
    dayyear_list = [1, 1]
    for day in range(365):
        for hour in range(24):
            if hour == 24:
                break
            else:
                day_list += [day + 1]

    df_smooth_UC['day'] = np.nan

    for index, row in df_smooth_UC.iterrows():
        day = day_list[index]
        df_smooth_UC.loc[index, 'day'] = day

    df_smooth_header = list(df_smooth_UC)

    df_day1_UC = df_smooth_UC.pivot_table(index=df_smooth_UC['day'],
                                          values=df_smooth_header[0:5],
                                          aggfunc='sum')

    df_day3_UC = df_day1_UC.append(df_day1_UC.append(df_day1_UC))
    df_day3_UC.reset_index(drop=True, inplace=True)

    # Export timeseries to SZN Linny-R
    df_day3_UC.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries daily input\Electricity demand_sznUC.txt",
        columns=['E demand'], header=None, index=None, sep=' ', mode='w')
    df_day3_UC.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries daily input\Hydrogen demand_full_sznUC.txt",
        columns=['H demand'], header=None, index=None, sep=' ', mode='w')
    df_day3_UC.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries daily input\Offshore windpark output_sznUC.txt",
        columns=['Offshore'], header=None, index=None, sep=' ', mode='w')
    df_day3_UC.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries daily input\Onshore windpark output _sznUC.txt",
        columns=['Onshore'], header=None, index=None, sep=' ', mode='w')
    df_day3_UC.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries daily input\Solar output one asset_sznUC.txt",
        columns=['Solar'], header=None, index=None, sep=' ', mode='w')

    # Run seasonal UC
    copy(r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Linny-R models\sznUC2030.lnr",
         r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver")
    timer = 0
    hard_reset = 0

    while not os.path.isfile(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\sznUC2030-data.txt"):  #this ensures script waits on Linny-R model
        sleep(1)
        timer += 1

    df_storage2050_UC = pd.read_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\sznUC2030-data.txt",
        sep='\t')

    os.remove(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\sznUC2030-data.txt")
    os.remove(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\sznUC2030-log.txt")
    os.remove(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\sznUC2030-stats.txt")

    # Import first year
    label1 = range(366, 549)
    df_storage_y2_UC = df_storage2050_UC.drop(labels=label1, axis=0)
    df_storage_y2_UC.reset_index(drop=True, inplace=True)

    # Make storage target every 3 days
    target_list = []
    target = 1

    for x in range(122):
        for day in range(72):
            if day == 71:
                target_list.append(target)
                target += 3
            else:
                target_list.append(0)

    df_rep_headers = ['LB Storage', 'UB Storage']
    df_storagetarget_UC_HSE = pd.DataFrame(columns=df_rep_headers)
    df_storagetarget_UC_HSCCGT = pd.DataFrame(columns=df_rep_headers)
    df_storagetarget_UC_HSOCGT = pd.DataFrame(columns=df_rep_headers)
    df_storagetarget1 = pd.DataFrame(columns=df_rep_headers)
    df_storagetarget2 = pd.DataFrame(columns=df_rep_headers)

    for index, row in df_UC2050_1.iterrows():
        target = target_list[index]
        if target > 0:
            df_storagetarget_UC_HSE.loc[index, 'LB Storage'] = df_storage_y2_UC.loc[
                target, 'Hydrogen Storage - Electrolyser S|L']
            df_storagetarget_UC_HSE.loc[index, 'UB Storage'] = df_storage_y2_UC.loc[
                target, 'Hydrogen Storage - Electrolyser S|L']
        else:
            df_storagetarget_UC_HSE.loc[index, 'LB Storage'] = 0
            df_storagetarget_UC_HSE.loc[index, 'UB Storage'] = 10e7

    for index, row in df_UC2050_1.iterrows():
        target = target_list[index]
        if target > 0:
            df_storagetarget_UC_HSCCGT.loc[index, 'LB Storage'] = df_storage_y2_UC.loc[
                target, 'Hydrogen Storage - Electrolyser CCGT|L']
            df_storagetarget_UC_HSCCGT.loc[index, 'UB Storage'] = df_storage_y2_UC.loc[
                target, 'Hydrogen Storage - Electrolyser CCGT|L']
        else:
            df_storagetarget_UC_HSCCGT.loc[index, 'LB Storage'] = 0
            df_storagetarget_UC_HSCCGT.loc[index, 'UB Storage'] = 10e7

    for index, row in df_UC2050_1.iterrows():
        target = target_list[index]
        if target > 0:
            df_storagetarget_UC_HSOCGT.loc[index, 'LB Storage'] = df_storage_y2_UC.loc[
                target, 'Hydrogen Storage - Electrolyser OCGT|L']
            df_storagetarget_UC_HSOCGT.loc[index, 'UB Storage'] = df_storage_y2_UC.loc[
                target, 'Hydrogen Storage - Electrolyser OCGT|L']
        else:
            df_storagetarget_UC_HSOCGT.loc[index, 'LB Storage'] = 0
            df_storagetarget_UC_HSOCGT.loc[index, 'UB Storage'] = 10e7

    # These are the storage targets at the beginning of this year for daily UC
    target_HSE = df_storage_y2_UC.loc[1, 'Hydrogen Storage - Electrolyser S|L']
    target_HSCCGT = df_storage_y2_UC.loc[1, 'Hydrogen Storage - Electrolyser CCGT|L']
    target_HSOCGT = df_storage_y2_UC.loc[1, 'Hydrogen Storage - Electrolyser OCGT|L']

    df_UC2050_1['target_HSE'] = np.where(df_UC2050_1['Timeseries'] >= 0, 0, 0)
    df_UC2050_1['target_HSCCGT'] = np.where(df_UC2050_1['Timeseries'] >= 0, 0, 0)
    df_UC2050_1['target_HSOCGT'] = np.where(df_UC2050_1['Timeseries'] >= 0, 0, 0)

    df_UC2050_1.loc[1, 'target_HSE'] = target_HSE
    df_UC2050_1.loc[1, 'target_HSCCGT'] = target_HSCCGT
    df_UC2050_1.loc[1, 'target_HSOCGT'] = target_HSOCGT

    # Export storage targets to Linny-R text files
    df_storagetarget_UC_HSE.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\LB_HSE.txt",
        columns=['LB Storage'], header=None, index=None, sep=' ', mode='w')
    df_storagetarget_UC_HSE.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\UB_HSE.txt",
        columns=['UB Storage'], header=None, index=None, sep=' ', mode='w')
    df_storagetarget_UC_HSCCGT.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\LB_HSCCGT.txt",
        columns=['LB Storage'], header=None, index=None, sep=' ', mode='w')
    df_storagetarget_UC_HSCCGT.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\UB_HSCCGT.txt",
        columns=['UB Storage'], header=None, index=None, sep=' ', mode='w')
    df_storagetarget_UC_HSOCGT.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\LB_HSOCGT.txt",
        columns=['LB Storage'], header=None, index=None, sep=' ', mode='w')
    df_storagetarget_UC_HSOCGT.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\UB_HSOCGT.txt",
        columns=['UB Storage'], header=None, index=None, sep=' ', mode='w')
    df_UC2050_1.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Initial levels\t0_target_HSE.txt",
        columns=['target_HSE'], header=None, index=None, sep=' ', mode='w')
    df_UC2050_1.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Initial levels\t0_target_HSCCGT.txt",
        columns=['target_HSCCGT'], header=None, index=None, sep=' ', mode='w')
    df_UC2050_1.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Initial levels\t0_target_HSOCGT.txt",
        columns=['target_HSOCGT'], header=None, index=None, sep=' ', mode='w')

    standard_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Standard_asset_output\_standard.txt"

    for assets in asset_capacity_dic:
        with open(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Standard_asset_output\_standard.txt",
                'w') as f:
            f.write(str(standard_capacity_dic[assets]))

        standard_exp_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Standard_asset_output" + "\\" + str(
            assets) + 'standard.txt'
        copy(standard_dir, standard_exp_dir)

    # Export yearly edemand growth to linny-r text files

    # Write input timeseries to Linny-R

    if year == 1:
        df_onshore = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Onshore windpark output _std_weather_full_year.txt",
            names=['Onshore'])
        df_offshore = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Offshore windpark output_std_weather_full_year.txt",
            names=['Offshore'])
        df_solar = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Solar output one asset_std_weather_full_year.txt",
            names=['Solar'])
        df_edemand = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Electricity demand_input.txt",
            names=['E demand'])
        df_hdemand = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Hydrogen demand_input.txt",
            names=['H demand'])
    else:
        df_onshore = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Onshore windpark output _std_weather_full_year.txt",
            names=['Onshore'])
        df_offshore = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Offshore windpark output_std_weather_full_year.txt",
            names=['Offshore'])
        df_solar = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Solar output one asset_std_weather_full_year.txt",
            names=['Solar'])
        df_edemand = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Electricity demand_full_year.txt",
            names=['E demand'])
        df_hdemand = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Hydrogen demand_full_year.txt",
            names=['H demand'])

    df_UC2050_1 = pd.concat([df_onshore, df_offshore, df_solar, df_edemand, df_hdemand], axis=1)

    df_UC2050_1['Timeseries'] = 1

    df_UC2050_1['Onshore'] = np.where(df_UC2050_1['Timeseries'] > 0,
                                      df_UC2050_1['Onshore'] * (asset_capacity_dic['OnshoreWind']), 0)
    df_UC2050_1['Offshore'] = np.where(df_UC2050_1['Timeseries'] > 0,
                                       df_UC2050_1['Offshore'] * (asset_capacity_dic['OffshoreWind']), 0)
    df_UC2050_1['Solar'] = np.where(df_UC2050_1['Timeseries'] > 0,
                                    df_UC2050_1['Solar'] * (asset_capacity_dic['SolarPV']),
                                    0)

    df_UC2050_1['E demand'] = np.where(df_UC2050_1['Timeseries'] > 0,
                                       df_UC2050_1['E demand'] * (electricity_demand_change),
                                       0)
    df_UC2050_1['H demand'] = np.where(df_UC2050_1['Timeseries'] > 0,
                                       df_UC2050_1['H demand'] * (hydrogen_demand_change), 0)

    df_UC2050_1.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Electricity demand_full_year.txt",
        columns=['E demand'], header=None, index=None, sep=' ', mode='w')
    df_UC2050_1.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Hydrogen demand_full_year.txt",
        columns=['H demand'], header=None, index=None, sep=' ', mode='w')
    df_UC2050_1.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Offshore windpark output_std_weather_full_year.txt",
        columns=['Offshore'], header=None, index=None, sep=' ', mode='w')
    df_UC2050_1.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Onshore windpark output _std_weather_full_year.txt",
        columns=['Onshore'], header=None, index=None, sep=' ', mode='w')
    df_UC2050_1.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Solar output one asset_std_weather_full_year.txt",
        columns=['Solar'], header=None, index=None, sep=' ', mode='w')

    # Run daily UC
    copy(r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Linny-R models\dailyUC2030.lnr",
         r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver")

    timer = 0

    while not os.path.isfile(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\dailyUC2030-data.txt"):
        sleep(1)
        timer += 1

    # These are the storage targets at the beginning of this year for daily UC
    target_HSE = df_storage_y2_UC.loc[365, 'Hydrogen Storage - Electrolyser S|L']
    target_HSCCGT = df_storage_y2_UC.loc[365, 'Hydrogen Storage - Electrolyser CCGT|L']
    target_HSOCGT = df_storage_y2_UC.loc[365, 'Hydrogen Storage - Electrolyser OCGT|L']

    df_UC2050_1['target_HSE'] = np.where(df_UC2050_1['Timeseries'] >= 0, 0, 0)
    df_UC2050_1['target_HSCCGT'] = np.where(df_UC2050_1['Timeseries'] >= 0, 0, 0)
    df_UC2050_1['target_HSOCGT'] = np.where(df_UC2050_1['Timeseries'] >= 0, 0, 0)

    df_UC2050_1.loc[1, 'target_HSE'] = target_HSE
    df_UC2050_1.loc[1, 'target_HSCCGT'] = target_HSCCGT
    df_UC2050_1.loc[1, 'target_HSOCGT'] = target_HSOCGT

    df_UC2050_1.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Initial levels\t0_target_HSE.txt",
        columns=['target_HSE'], header=None, index=None, sep=' ', mode='w')
    df_UC2050_1.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Initial levels\t0_target_HSCCGT.txt",
        columns=['target_HSCCGT'], header=None, index=None, sep=' ', mode='w')
    df_UC2050_1.to_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Initial levels\t0_target_HSOCGT.txt",
        columns=['target_HSOCGT'], header=None, index=None, sep=' ', mode='w')

    df_linny = pd.read_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\dailyUC2030-data.txt",
        sep='\t', skiprows=[1])

    os.remove(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\dailyUC2030-data.txt")
    os.remove(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\dailyUC2030-log.txt")
    os.remove(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\dailyUC2030-stats.txt")

    df_linny.columns = df_linny.columns.str.replace(' ', '', regex=True)
    df_linny.columns = df_linny.columns.str.replace('|', '', regex=True)

    # Add e-price to linny import
    asset_MC_dic_sorted = sorted(asset_MC_dic.items(), key=lambda x: x[1], reverse=True)

    counter = 0

    for index, column in df_linny.iterrows():
        for assets in asset_MC_dic_sorted:
            if investment_list_classifier_dic[assets[0]] != 'Storage':
                if df_linny.loc[index, asset_mc_linny_output_dic[assets[0]]] > 0:
                    E_price_during_t = assets[1]
                    df_linny.loc[index, 'e-price'] = E_price_during_t
                    counter += 1
                    break

    df_linny['e-price'] = np.where(df_linny['VOLLL'] > 0, VOLL, df_linny['e-price'])
    df_linny['e-price'] = np.where(df_linny['HydrogenimportL'] > 0, VOLL, df_linny['e-price'])

    df_onshore = pd.read_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Onshore windpark output _std_weather_full_year.txt",
        names=['Onshore'])
    df_offshore = pd.read_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Offshore windpark output_std_weather_full_year.txt",
        names=['Offshore'])
    df_solar = pd.read_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Solar output one asset_std_weather_full_year.txt",
        names=['Solar'])

    df_tot_data_UC = pd.concat([df_linny, df_offshore, df_onshore, df_solar], axis=1)

    # Calculate REV per asset

    for asset_groups in investment_list_dic:
        remove1 = len(asset_groups) - len('_asset_list')
        if investment_list_classifier_dic[asset_groups[:remove1]] == 'Asset':
            max_output_asset = standard_capacity_dic[asset_groups[:remove1]]
            for individual_assets in investment_list_dic[asset_groups]:
                if all_asset_capacity_dic[individual_assets] == 1:
                    df_tot_data_UC[individual_assets + '_output'] = np.where(
                        df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]] - max_output_asset > 0,
                        max_output_asset, df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]])

                    df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]] = np.where(
                        df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]] - max_output_asset > 0,
                        df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]] - max_output_asset, 0)

                    df_tot_data_UC[individual_assets] = np.where(df_tot_data_UC[individual_assets + '_output'] > 0,
                                                                 df_tot_data_UC[individual_assets + '_output'] * (
                                                                         df_tot_data_UC['e-price'] - asset_MC_dic[
                                                                     asset_groups[:remove1]]), 0)

        if investment_list_classifier_dic[asset_groups[:remove1]] == 'VRES':

            for individual_assets in investment_list_dic[asset_groups]:

                if all_asset_capacity_dic[individual_assets] == 1:
                    df_tot_data_UC[individual_assets + '_output'] = np.where(
                        df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]] - df_tot_data_UC[
                            weather_dic[asset_groups[:remove1]]] > 0,
                        df_tot_data_UC[weather_dic[asset_groups[:remove1]]],
                        df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]])
                    df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]] = np.where(
                        df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]] - df_tot_data_UC[
                            weather_dic[asset_groups[:remove1]]] > 0,
                        df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]] - df_tot_data_UC[
                            weather_dic[asset_groups[:remove1]]], 0)
                    df_tot_data_UC[individual_assets] = np.where(df_tot_data_UC[individual_assets + '_output'] > 0,
                                                                 df_tot_data_UC[individual_assets + '_output'] *
                                                                 df_tot_data_UC['e-price'], 0)

        if investment_list_classifier_dic[asset_groups[:remove1]] == 'Storage':

            if storage_list_classifier_dic[asset_groups[:remove1]] == 'Hydrogen':

                for individual_assets in investment_list_dic[asset_groups]:
                    if all_asset_capacity_dic[individual_assets] == 1:

                        if charge_storage_dic[investment_storage_dic[asset_groups[:remove1]][0]] == 'Charge':
                            charge_asset = investment_storage_dic[asset_groups[:remove1]][0]
                            max_output_charge_asset = standard_capacity_dic[charge_asset]

                            df_tot_data_UC[individual_assets + '_Charge'] = np.where(
                                df_tot_data_UC[asset_linny_output_dic[charge_asset]] - max_output_charge_asset > 0,
                                max_output_charge_asset, df_tot_data_UC[asset_linny_output_dic[charge_asset]])
                            df_tot_data_UC[asset_linny_output_dic[charge_asset]] = np.where(
                                df_tot_data_UC[asset_linny_output_dic[charge_asset]] - max_output_charge_asset > 0,
                                df_tot_data_UC[asset_linny_output_dic[charge_asset]] - max_output_charge_asset, 0)
                            df_tot_data_UC[individual_assets + '_ChargeRev'] = np.where(
                                df_tot_data_UC[individual_assets + '_Charge'] > 0,
                                df_tot_data_UC[individual_assets + '_Charge'] * (df_tot_data_UC['e-price'] - 3), 0)

                        if discharge_storage_dic[
                            investment_storage_dic[asset_groups[:remove1]][1]] != 'ElectrolyserSDischarge':

                            discharge_asset = investment_storage_dic[asset_groups[:remove1]][1]
                            max_output_discharge_asset = standard_capacity_dic[discharge_asset]

                            if discharge_storage_dic[
                                investment_storage_dic[asset_groups[:remove1]][1]] == 'Discharge':
                                df_tot_data_UC[individual_assets + '_Discharge'] = np.where(df_tot_data_UC[
                                                                                                asset_linny_output_dic[
                                                                                                    discharge_asset]] - max_output_discharge_asset > 0,
                                                                                            max_output_discharge_asset,
                                                                                            df_tot_data_UC[
                                                                                                asset_linny_output_dic[
                                                                                                    discharge_asset]])
                                df_tot_data_UC[asset_linny_output_dic[discharge_asset]] = np.where(df_tot_data_UC[
                                                                                                       asset_linny_output_dic[
                                                                                                           discharge_asset]] - max_output_discharge_asset > 0,
                                                                                                   df_tot_data_UC[
                                                                                                       asset_linny_output_dic[
                                                                                                           discharge_asset]] - max_output_discharge_asset,
                                                                                                   0)
                                df_tot_data_UC[individual_assets + '_DischargeRev'] = np.where(
                                    df_tot_data_UC[individual_assets + '_Discharge'] > 0,
                                    df_tot_data_UC[individual_assets + '_Discharge'] * (df_tot_data_UC['e-price']),
                                    0)

                        else:
                            discharge_asset = investment_storage_dic[asset_groups[:remove1]][1]
                            max_output_discharge_asset = standard_capacity_dic[discharge_asset]

                            if discharge_storage_dic[
                                investment_storage_dic[asset_groups[:remove1]][1]] == 'Discharge':
                                df_tot_data_UC[individual_assets + '_Discharge'] = np.where(
                                    df_tot_data_UC[asset_linny_output_dic[discharge_asset]] > 0,
                                    df_tot_data_UC[asset_linny_output_dic[discharge_asset]], 0)
                                df_tot_data_UC[individual_assets + '_DischargeRev'] = np.where(
                                    df_tot_data_UC[individual_assets + '_Discharge'] > 0,
                                    df_tot_data_UC[individual_assets + '_Discharge'] * (df_tot_data_UC['e-price']),
                                    0)
            else:
                for individual_assets in investment_list_dic[asset_groups]:
                    if all_asset_capacity_dic[individual_assets] == 1:

                        if charge_storage_dic[investment_storage_dic[asset_groups[:remove1]][0]] == 'Charge':
                            charge_asset = investment_storage_dic[asset_groups[:remove1]][0]
                            max_output_charge_asset = standard_capacity_dic[charge_asset]

                            df_tot_data_UC[individual_assets + '_Charge'] = np.where(
                                df_tot_data_UC[asset_linny_output_dic[charge_asset]] - max_output_charge_asset > 0,
                                max_output_charge_asset, df_tot_data_UC[asset_linny_output_dic[charge_asset]])
                            df_tot_data_UC[asset_linny_output_dic[charge_asset]] = np.where(
                                df_tot_data_UC[asset_linny_output_dic[charge_asset]] - max_output_charge_asset > 0,
                                df_tot_data_UC[asset_linny_output_dic[charge_asset]] - max_output_charge_asset, 0)
                            df_tot_data_UC[individual_assets + '_ChargeRev'] = np.where(
                                df_tot_data_UC[individual_assets + '_Charge'] > 0,
                                df_tot_data_UC[individual_assets + '_Charge'] * df_tot_data_UC['e-price'], 0)

                        if discharge_storage_dic[
                            investment_storage_dic[asset_groups[:remove1]][1]] != 'ElectrolyserSDischarge':

                            discharge_asset = investment_storage_dic[asset_groups[:remove1]][1]
                            max_output_discharge_asset = standard_capacity_dic[discharge_asset]

                            if discharge_storage_dic[
                                investment_storage_dic[asset_groups[:remove1]][1]] == 'Discharge':
                                df_tot_data_UC[individual_assets + '_Discharge'] = np.where(df_tot_data_UC[
                                                                                                asset_linny_output_dic[
                                                                                                    discharge_asset]] - max_output_discharge_asset > 0,
                                                                                            max_output_discharge_asset,
                                                                                            df_tot_data_UC[
                                                                                                asset_linny_output_dic[
                                                                                                    discharge_asset]])
                                df_tot_data_UC[asset_linny_output_dic[discharge_asset]] = np.where(df_tot_data_UC[
                                                                                                       asset_linny_output_dic[
                                                                                                           discharge_asset]] - max_output_discharge_asset > 0,
                                                                                                   df_tot_data_UC[
                                                                                                       asset_linny_output_dic[
                                                                                                           discharge_asset]] - max_output_discharge_asset,
                                                                                                   0)
                                df_tot_data_UC[individual_assets + '_DischargeRev'] = np.where(
                                    df_tot_data_UC[individual_assets + '_Discharge'] > 0,
                                    df_tot_data_UC[individual_assets + '_Discharge'] * df_tot_data_UC['e-price'], 0)

                        else:
                            discharge_asset = investment_storage_dic[asset_groups[:remove1]][1]
                            max_output_discharge_asset = standard_capacity_dic[discharge_asset]

                            if discharge_storage_dic[
                                investment_storage_dic[asset_groups[:remove1]][1]] == 'Discharge':
                                df_tot_data_UC[individual_assets + '_Discharge'] = np.where(
                                    df_tot_data_UC[asset_linny_output_dic[discharge_asset]] > 0,
                                    df_tot_data_UC[asset_linny_output_dic[discharge_asset]], 0)
                                df_tot_data_UC[individual_assets + '_DischargeRev'] = np.where(
                                    df_tot_data_UC[individual_assets + '_Discharge'] > 0,
                                    df_tot_data_UC[individual_assets + '_Discharge'] * df_tot_data_UC['e-price'], 0)

    # Determine CF for all assets - UC (CF = previous rev - fix OM)
    df_asset_cf_UC = pd.DataFrame()

    for asset_groups in investment_list_dic:

        remove1 = len(asset_groups) - len('_asset_list')

        if investment_list_classifier_dic[asset_groups[:remove1]] == 'Asset':
            for individual_assets in investment_list_dic[asset_groups]:
                if all_asset_capacity_dic[individual_assets] == 1:
                    CF = df_tot_data_UC[individual_assets].sum() - asset_OM_dic[asset_groups[:remove1]]
                    df_asset_cf_UC.loc[year, individual_assets] = CF

        elif investment_list_classifier_dic[asset_groups[:remove1]] == 'VRES':
            for individual_assets in investment_list_dic[asset_groups]:
                if all_asset_capacity_dic[individual_assets] == 1:
                    CF = df_tot_data_UC[individual_assets].sum() - asset_OM_dic[asset_groups[:remove1]]
                    df_asset_cf_UC.loc[year, individual_assets] = CF

        elif investment_list_classifier_dic[asset_groups[:remove1]] == 'Storage':
            for individual_assets in investment_list_dic[asset_groups]:
                if all_asset_capacity_dic[individual_assets] == 1:
                    CF_charge = df_tot_data_UC[individual_assets + '_ChargeRev'].sum()
                    CF_discharge = df_tot_data_UC[individual_assets + '_DischargeRev'].sum()
                    CF = (CF_discharge - CF_charge) - asset_OM_dic[asset_groups[:remove1]]
                    df_asset_cf_UC.loc[year, individual_assets] = CF

    # D = df_UC2050_1['E demand'].max() + df_UC2050_1['H demand'].max()
    Epeak_y4 = df_UC2050_1['E demand'].max()
    Hpeak_y4 = df_UC2050_1['H demand'].max()
    Dpeak = Epeak_y4 + Hpeak_y4
    Dr = (Dpeak) * (1 + r)

    if capacity_mechanism == 1:

        df_asset_cm = pd.DataFrame()
        df_asset_cm_sorted = pd.DataFrame()

        counter = 0
        cm_assets = []

        for asset_groups in investment_list_dic:
            remove1 = len(asset_groups) - len('_asset_list')
            for individual_assets in investment_list_dic[asset_groups]:
                if all_asset_capacity_dic[individual_assets] == 1:
                    if df_asset_cf_UC.loc[year, individual_assets] < 0:
                        counter += 1
                        cm_assets += [individual_assets]
                        df_asset_cm.loc[individual_assets, 'Volume bid'] = standard_capacity_dic[
                            asset_groups[:remove1]]
                        df_asset_cm.loc[individual_assets, 'Price bid'] = (df_asset_cf_UC.loc[
                                                                               year, individual_assets] * -1) / (
                                                                              standard_capacity_dic[
                                                                                  asset_groups[:remove1]])
                        df_asset_cm_sorted = df_asset_cm.sort_values('Price bid')
        tot_volume = 0

        for asset in range(len(df_asset_cm_sorted)):
            tot_volume += df_asset_cm_sorted.iloc[asset][0]
            df_asset_cm_sorted.iloc[asset][0] = tot_volume

        if len(df_asset_cm_sorted) < 2:
            if len(df_asset_cm_sorted) == 0:
                df_asset_cm_sorted['Volume bid'] = 0
                df_asset_cm_sorted['Price bid'] = 0
            else:
                df_asset_cm_sorted = df_asset_cm_sorted.append(df_asset_cm_sorted.iloc[0])
                df_asset_cm_sorted.iloc[0, 0] = 0
                df_asset_cm_sorted.iloc[0, 1] = df_asset_cm_sorted.iloc[1, 1]

        # df_asset_cm_sorted.plot(x='Volume bid', y='Price bid')
        # plt.ylabel('€/MWh')
        # plt.xlabel('MW')
        # plt.title('Capacity market bids in year' + str(year))
        # plt.show()

        if len(df_asset_cm_sorted) > 0:
            if len(df_asset_cm_sorted) > 2:

                # Input variables
                demand_list = []
                t_list = []
                counter = 0
                LM = 0
                UM = 0
                m = 0
                x = 0
                y = 0

                # Calculations
                UM = (Dpeak * (1 + r + um_in))
                LM = (Dpeak * (1 + r - lm_in))
                # m = -1 * (Pc / (LM - UM))  # slope
                # b = (m * UM)

                a = 0
                b = 0

                a = (0 - Pc) / (UM - LM)
                b = Pc - (a * LM)

                volume1 = df_asset_cm_sorted['Volume bid']
                volume2 = np.linspace(0, LM, 500)
                volume3 = np.linspace(LM, UM, 500)

                m1, b1 = 0, Pc
                m2, b2 = a, b

                price1 = df_asset_cm_sorted['Price bid']
                price2 = (volume2 * m1) + b1
                price3 = (volume3 * m2) + b2

                line_1 = LineString(np.column_stack((volume1, price1)))
                line_2 = LineString(np.column_stack((volume2, price2)))
                line_3 = LineString(np.column_stack((volume3, price3)))

                volume_intersection = 0
                price_intersection = 0

                intersection = line_1.intersection(line_3)

                if intersection.is_empty:
                    intersection = Point(0, 0)

                    intersection = line_1.intersection(line_2)
                    if intersection.is_empty:
                        intersection = Point(0, 0)

                        if df_asset_cm_sorted['Price bid'].min() < Pc:
                            for index, row in df_asset_cm_sorted.iterrows():
                                if df_asset_cm_sorted.loc[index, 'Price bid'] <= Pc:
                                    volume_intersection = volume_intersection + df_asset_cm_sorted.loc[
                                        index, 'Volume bid']
                                    price_intersection = df_asset_cm_sorted.loc[index, 'Price bid']
                                    if volume_intersection <= Dr:
                                        volume_intersection_actual = volume_intersection
                                        intersection = Point(volume_intersection_actual, price_intersection)

                x, y = intersection.xy

                CM_volume = x[0]
                CM_price = y[0]

                if CM_volume > 0:
                    if CM_price > 0:
                        total_CM_costs = CM_price * CM_volume

                # plt.plot(volume1, price1, 'blue')
                # plt.plot(volume2, price2, 'black')
                # plt.plot(volume3, price3, 'black')
                # plt.plot(x, y, 'ro')
                # plt.ylim(0, Pc * 1.1)
                # plt.title('Capacity market in year' + str(year))
                # plt.xlabel('MW')
                # plt.ylabel('€/MWh')
                # plt.show()

                max_cm = x[0]

                df_cm_cf = pd.DataFrame()

                for index, row in df_asset_cm_sorted.iterrows():
                    if max_cm - df_asset_cm['Volume bid'][index] > 0:
                        max_cm = max_cm - df_asset_cm['Volume bid'][index]
                        CF = df_asset_cm['Volume bid'][index] * y[0]
                        df_cm_cf.loc[year, index] = CF
                    elif max_cm > 0:
                        CF = max_cm * y[0]
                        df_cm_cf.loc[year, index] = CF
                        max_cm = 0

                for name, values in df_cm_cf.iteritems():
                    df_asset_cf_UC.loc[year, name] = df_asset_cf_UC.loc[year, name] + df_cm_cf.loc[year, name]

                df_dismantle = df_dismantle.append(df_asset_cf_UC)
            else:
                # Input variables
                demand_list = []
                t_list = []
                counter = 0
                LM = 0
                UM = 0
                m = 0
                x = 0
                y = 0

                # Calculations
                UM = (Dpeak * (1 + r + um_in))
                LM = (Dpeak * (1 + r - lm_in))
                # m = -1 * (Pc / (LM - UM))  # slope
                # b = (m * UM)

                a = 0
                b = 0

                a = (0 - Pc) / (UM - LM)
                b = Pc - (a * LM)

                volume1 = df_asset_cm_sorted['Volume bid']
                volume2 = np.linspace(0, LM, 500)
                volume3 = np.linspace(LM, UM, 500)

                m1, b1 = 0, Pc
                m2, b2 = a, b

                price1 = df_asset_cm_sorted['Price bid']
                price2 = (volume2 * m1) + b1
                price3 = (volume3 * m2) + b2

                line_1 = LineString(np.column_stack((volume1, price1)))
                line_2 = LineString(np.column_stack((volume2, price2)))
                line_3 = LineString(np.column_stack((volume3, price3)))

                volume_intersection = 0
                price_intersection = 0

                intersection = line_1.intersection(line_3)

                if intersection.is_empty:
                    intersection = Point(0, 0)

                    intersection = line_1.intersection(line_2)
                    if intersection.is_empty:
                        intersection = Point(0, 0)

                        if df_asset_cm_sorted['Price bid'].min() < Pc:
                            for row in range(len(df_asset_cm_sorted.index)):
                                if df_asset_cm_sorted.iloc[row, 1] <= Pc:
                                    volume_intersection = df_asset_cm_sorted.iloc[row, 0]
                                    if volume_intersection <= Dr:
                                        volume_intersection_actual = volume_intersection
                                        price_intersection = df_asset_cm_sorted.iloc[row, 1]
                                        intersection = Point(volume_intersection_actual, price_intersection)

                x, y = intersection.xy

                CM_volume = x[0]
                CM_price = y[0]

                if CM_volume > 0:
                    if CM_price > 0:
                        total_CM_costs = CM_price * CM_volume

                # plt.plot(volume1, price1, 'blue')
                # plt.plot(volume2, price2, 'black')
                # plt.plot(volume3, price3, 'red')
                # plt.plot(x, y, 'ro')
                # plt.ylim(0, Pc * 1.1)
                # plt.title('Capacity market in year' + str(year))
                # plt.xlabel('MW')
                # plt.ylabel('€/MWh')

                plt.show()

                max_cm = x[0]

                df_cm_cf = pd.DataFrame()

                for index, row in df_asset_cm_sorted.iterrows():
                    if max_cm - df_asset_cm['Volume bid'][index] > 0:
                        max_cm = max_cm - df_asset_cm['Volume bid'][index]
                        CF = df_asset_cm['Volume bid'][index] * y[0]
                        df_cm_cf.loc[year, index] = CF
                    elif max_cm > 0:
                        CF = max_cm * y[0]
                        df_cm_cf.loc[year, index] = CF
                        max_cm = 0

                for name, values in df_cm_cf.iteritems():
                    df_asset_cf_UC.loc[year, name] = df_asset_cf_UC.loc[year, name] + df_cm_cf.loc[year, name]

                df_dismantle = df_dismantle.append(df_asset_cf_UC)

    elif capacity_mechanism == 0:
        df_dismantle = df_dismantle.append(df_asset_cf_UC)

    # Create_lists_containing_all_assets to determine what assets are
    dismantle_list_dic = cpy.deepcopy(investment_list_dic)

    for asset_groups in dismantle_list_dic:
        for individual_assets in dismantle_list_dic[asset_groups]:
            all_asset_age_dic[individual_assets] = all_asset_age_dic[individual_assets] + 1

    # Export all relevant values from the daily-UC run

    total_capacity = 0
    supply_ratio = 0

    for asset_groups in investment_list_dic:
        remove1 = len(asset_groups) - len('_asset_list')
        if investment_list_classifier_dic[asset_groups[:remove1]] == 'Asset':
            standard_asset = standard_capacity_dic[asset_groups[:remove1]]
            asset_capacity = asset_capacity_dic[asset_groups[:remove1]]
            total_capacity += (standard_asset * asset_capacity)
        if investment_list_classifier_dic[asset_groups[:remove1]] == 'Storage':
            standard_asset = standard_capacity_dic[asset_groups[:remove1]]
            asset_capacity = asset_capacity_dic[asset_groups[:remove1]]

            if asset_groups == 'LiOnStorage_asset_list':
                total_capacity += (standard_asset * 0.42) * (asset_capacity)
            if asset_groups == 'ElectrolyserS_asset_list':
                total_capacity += (standard_asset * 0.7) * (asset_capacity)
            if asset_groups == 'ElectrolyserCCGT_asset_list':
                total_capacity += (standard_asset * 0.57 * asset_capacity)
            if asset_groups == 'ElectrolyserOCGT_asset_list':
                total_capacity += (standard_asset * 0.39 * asset_capacity)

    df_dismantle_min = cpy.deepcopy(df_UC2050_1)
    df_UC2050_1["Residual load"] = np.where(df_UC2050_1['Timeseries'] > 0,
                                            (df_UC2050_1['H demand'] + df_UC2050_1['E demand']) - (
                                                        df_UC2050_1['Onshore'] + df_UC2050_1['Offshore'] + df_UC2050_1[
                                                    'Solar']), 0)
    D_residual = df_UC2050_1["Residual load"].max()
    supply_ratio = (total_capacity) / D_residual

    df_linny['VOLL'] = np.where(df_linny['e-price'] == VOLL,
                                1, 0)

    shortage_volume = df_linny['VOLLL'].sum()

    shortage_hours = df_linny['VOLL'].sum()

    mean_e_price = df_linny['e-price'].mean()

    df_linny['DemandCost'] = np.where(df_linny['t'] > 1,
                                      (df_linny['electricitydemandL'] * df_linny['e-price']) +
                                      (df_linny['hydrogendemandL'] * df_linny['e-price']), 0)

    demand_costs = df_linny['DemandCost'].sum()

    if capacity_mechanism == 0:
        total_CM_costs = 0
        CM_volume = 0
        CM_price = 0

    total_costs = total_CM_costs + demand_costs

    df_key_indicators.iloc[year - 1, 0] = supply_ratio
    df_key_indicators.iloc[year - 1, 1] = shortage_volume
    df_key_indicators.iloc[year - 1, 2] = shortage_hours
    df_key_indicators.iloc[year - 1, 3] = mean_e_price
    df_key_indicators.iloc[year - 1, 4] = CM_volume
    df_key_indicators.iloc[year - 1, 5] = CM_price
    df_key_indicators.iloc[year - 1, 6] = total_CM_costs
    df_key_indicators.iloc[year - 1, 7] = total_costs

    for assets in asset_capacity_dic:
        df_merit_order.loc[year, str(assets)] = asset_capacity_dic[assets]

    if year + 1 > end_year:
        if capacity_mechanism == 1:
            df_key_indicators['Experiment'] = experiment
            df_key_indicators['CM'] = 1
            df_key_indicators['Value year'] = 1
            df_merit_order['Experiment'] = experiment
            df_merit_order['CM'] = 1
            df_merit_order['Value year'] = 1

            for value_year in range(end_year):
                df_key_indicators.iloc[value_year, 10] = (value_year + 1)

            for value_year in range(end_year):
                df_merit_order.iloc[value_year, 14] = (value_year + 1)

            df_key_indicators.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Output data\key_indicators.csv",
                index=False)
            keyindicator_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Output data\key_indicators.csv"
            keyindicator_exp_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Output data\key_indicators.csv" + '_CM_' + str(
                experiment) + '.csv'

            df_merit_order.to_csv(r"C:\Users\IEUser\Desktop\Yasin thesi files\Output data\merit_order.csv",
                                  index=False)
            merit_order_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Output data\merit_order.csv"
            mertit_order_exp_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Output data\merit_order.csv" + "_CM_" + str(
                experiment) + ".csv"

        if capacity_mechanism == 0:
            df_key_indicators['Experiment'] = experiment
            df_key_indicators['CM'] = 0
            df_key_indicators['Value year'] = 1
            df_merit_order['Experiment'] = experiment
            df_merit_order['CM'] = 0
            df_merit_order['Value year'] = 1

            for value_year in range(end_year):
                df_key_indicators.iloc[value_year, 10] = (value_year + 1)

            for value_year in range(end_year):
                df_merit_order.iloc[value_year, 14] = (value_year + 1)

            df_key_indicators.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Output data\key_indicators.csv",
                index=False)
            keyindicator_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Output data\key_indicators.csv"
            keyindicator_exp_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Output data\key_indicators.csv" + '_NOCM_' + str(
                experiment) + '.csv'

            df_merit_order.to_csv(r"C:\Users\IEUser\Desktop\Yasin thesi files\Output data\merit_order.csv",
                                  index=False)
            merit_order_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Output data\merit_order.csv"
            mertit_order_exp_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Output data\merit_order.csv" + "_NOCM_" + str(
                experiment) + ".csv"

        copy(keyindicator_dir, keyindicator_exp_dir)
        copy(merit_order_dir, mertit_order_exp_dir)

    ##############################################################################################
    #                                                                                            #
    #                          #CODE TO RUN THE FP PART OF MIDO MODEL                            #
    #                                                                                            #
    ##############################################################################################

    budget = sum(df_asset_cf_UC.sum(1))

    # if budget > 0:
    #     budget = budget
    # else:
    #     budget = 0

    if year == 1:
        budget_number = budget + budget_year0
        budget_number = budget_number - df_debt.loc[year].sum()
    else:
        budget_number += budget
        tot_debt = sum(df_debt.sum(1))
        if tot_debt > 0:
            budget_number = budget_number - df_debt.loc[year].sum()

    negative_npv = 0
    investment_round = 0

    assets_this_round = []

    # Update prices to match FP loop ahead year
    FP_CO2_price = cpy.deepcopy(CO2_price)
    FP_biomass_price = cpy.deepcopy(biomass_price)
    FP_hydrogen_import_price = cpy.deepcopy(hydrogen_import_price)
    FP_gas_price = cpy.deepcopy(gas_price)
    FP_uranium_price = cpy.deepcopy(uranium_price)

    if year < 14:
        FP_CO2_price = FP_CO2_price * (CO2_development ** (fp_lookahead))
        FP_biomass_price = FP_biomass_price * (biomass_development ** (fp_lookahead))
        FP_hydrogen_import_price = FP_hydrogen_import_price * (hydrogen_development ** (fp_lookahead))
        FP_gas_price = FP_gas_price * (gas_development ** (fp_lookahead))
        FP_uranium_price = FP_uranium_price * (uranium_development ** (fp_lookahead))

    FP_asset_only_MC_dic = {'Free': 0, 'Biomass': FP_biomass_price, 'Gas': FP_gas_price, 'Uranium': FP_uranium_price,
                            'Hydrogen_import': FP_hydrogen_import_price}

    # Update investment costs to reflect reference years
    FP_euro_mw_dic = cpy.deepcopy(euro_mw_dic)
    FP_investment_costs_dic = {}
    FP_asset_OM_dic = {}

    if year < 14:
        for assets in investment_list:
            if assets == 'OffshoreWind':
                FP_euro_mw_dic[assets] = euro_mw_dic[assets] * ((1 + offshore_development) ** (fp_lookahead))
            if assets == 'OnshoreWind':
                FP_euro_mw_dic[assets] = euro_mw_dic[assets] * ((1 + onshore_development) ** (fp_lookahead))
            if assets == 'SolarPV':
                FP_euro_mw_dic[assets] = euro_mw_dic[assets] * ((1 + solar_development) ** (fp_lookahead))
            if assets == 'GasCCGT':
                FP_euro_mw_dic[assets] = euro_mw_dic[assets] * ((1 + gas_CCGT_development) ** (fp_lookahead))
            if assets == 'ElectrolyserS':
                FP_euro_mw_dic[assets] = euro_mw_dic[assets] * ((1 + electrolyser_development) ** (fp_lookahead))
            if assets == 'ElectrolyerCCGT':
                FP_euro_mw_dic[assets] = 0.945 + (euro_mw_dic[assets] - 0.945) * (
                            (1 + electrolyser_development) ** (fp_lookahead))
            if assets == 'ElectrolyerOCGT':
                FP_euro_mw_dic[assets] = 0.420 + (euro_mw_dic[assets] - 0.420) * (
                            (1 + electrolyser_development) ** (fp_lookahead))
            if assets == 'LiOnStorage':
                FP_euro_mw_dic[assets] = euro_mw_dic[assets] * ((1 + storage_development) ** (fp_lookahead))

    for assets in investment_list:
        FP_investment_costs_dic[assets] = standard_capacity_dic[assets] * FP_euro_mw_dic[assets] * 1e6

    key_min = min(FP_investment_costs_dic.keys(), key=(lambda k: FP_investment_costs_dic[k]))
    min_cost = FP_investment_costs_dic[key_min]

    for assets in investment_list:
        FP_asset_OM_dic[assets] = FP_investment_costs_dic[assets] * fix_OM_per_dic[assets]

    # Write MC all assets to Linny-R

    for assets in investment_list:
        CO2_costs = (CO2_dic[assets] * FP_CO2_price)
        if investment_list_classifier_dic[assets] != 'Storage':
            if fuel_search_dic[assets] == 'Gas':
                fuel_costs = FP_asset_only_MC_dic[fuel_search_dic[assets]] / gas_asset_efficiency[assets]
            elif fuel_search_dic[assets] == 'Uranium':
                fuel_costs = FP_asset_only_MC_dic[fuel_search_dic[assets]] / nuclear_asset_effiency[assets]
            elif fuel_search_dic[assets] == 'Biomass':
                fuel_costs = FP_asset_only_MC_dic[fuel_search_dic[assets]] / biomass_asset_effiency[assets]
            else:
                fuel_costs = 0

            MC = CO2_costs + fuel_costs

            asset_MC_dic[assets] = MC
        else:
            asset_MC_dic[assets] = 0

    MC_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Marginal costs_output\MC.txt"

    for assets in investment_list:
        with open(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Marginal costs_output\MC.txt",
                'w') as f:
            f.write(str(asset_MC_dic[assets]))

        MC_exp_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Marginal costs_output" + "\\" + str(
            assets) + 'MC.txt'

        copy(MC_dir, MC_exp_dir)

    # Update e_demand an h_demand to reflect reference year

    df_edemand = pd.read_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Electricity demand_full_year.txt",
        names=['E demand'])

    df_hdemand = pd.read_csv(
        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Hydrogen demand_full_year.txt",
        names=['H demand'])

    df_edemand['Timeseries'] = 1
    df_hdemand['Timeseries'] = 1

    if year < 14:
        df_edemand['E demand'] = np.where(df_edemand['Timeseries'] > 0,
                                          df_edemand['E demand'] * (electricity_demand_change ** fp_lookahead), 0)
        df_hdemand['H demand'] = np.where(df_hdemand['Timeseries'] > 0,
                                          df_hdemand['H demand'] * (hydrogen_demand_change ** fp_lookahead), 0)
    df_edemand = df_edemand.drop(['Timeseries'], axis=1)
    df_hdemand = df_hdemand.drop(['Timeseries'], axis=1)

    iteration = 0

    while (budget_number > 0) and (negative_npv == 0) and (
            ((budget_number / energy_companies)) >= min_cost * equity_factor):

        investment_round += 1

        # Copy all relevant dictionaries
        FP_delay_dic = cpy.deepcopy(delay_dic)
        all_asset_age_dic_copy = cpy.deepcopy(all_asset_age_dic)
        all_asset_capacity_dic_copy = cpy.deepcopy(all_asset_capacity_dic)
        FP_asset_capacity_dic_copy = cpy.deepcopy(asset_capacity_dic)
        delay_asset_capacity_dic_copy = cpy.deepcopy(asset_capacity_dic)
        installed_counter_dic_copy = cpy.deepcopy(installed_counter_dic)
        FP_installed_counter_dic = cpy.deepcopy(installed_counter_dic)
        delay_installed_counter_dic = cpy.deepcopy(installed_counter_dic)
        FP_pipeline_list_dic = cpy.deepcopy(pipeline_list_dic)

        FP_all_asset_capacity_dic = {}
        FP_all_asset_capacity_dic_nocd = {}
        FP_VRES_output_dic = {}
        FP_storage_charge_discharge_dic = {}
        FP_combined_storage_list = []

        # Remove all assets that need to be dismantled due to age from merit order or add assets that are in pipeline/from this round

        FP_delay_asset_capacity_dic = {}
        FP_asset_this_round_dic = {}

        FP_year_range = range(cpy.deepcopy(year), (cpy.deepcopy(year) + fp_lookahead))

        # Remove assets that need to be dismantled due to age from merit order
        for asset_groups in dismantle_list_dic:
            remove1 = len(asset_groups) - len('_asset_list')
            for individual_assets in dismantle_list_dic[asset_groups]:
                all_asset_age_dic_copy[individual_assets] = all_asset_age_dic_copy[individual_assets] + fp_lookahead
                if asset_lifetime_dic[asset_groups[:remove1]] <= all_asset_age_dic_copy[individual_assets]:
                    all_asset_capacity_dic_copy[individual_assets] = 0
                    FP_asset_capacity_dic_copy[asset_groups[:remove1]] = FP_asset_capacity_dic_copy[
                                                                             asset_groups[:remove1]] - 1
                    delay_asset_capacity_dic_copy[asset_groups[:remove1]] = delay_asset_capacity_dic_copy[
                                                                                asset_groups[:remove1]] - 1
                    # FP_installed_counter_dic[asset_groups[:remove1] + '_installed_counter'] = \
                    # FP_installed_counter_dic[
                    #     asset_groups[
                    #     :remove1] + '_installed_counter'] - 1
                    delay_installed_counter_dic[asset_groups[:remove1] + '_installed_counter'] = \
                        delay_installed_counter_dic[asset_groups[:remove1] + '_installed_counter'] - 1

        assets_out_pipeline_list = []
        # Add assets that are in pipeline to merit order
        assets_this_round_skip = 0
        for FP_year in FP_year_range:
            assets_out_pipeline_list += getKeysByValue(FP_delay_dic, FP_year)

        for asset_groups in investment_list_dic:
            remove1 = len(asset_groups) - len('_asset_list')
            for individual_assets in assets_out_pipeline_list:
                if individual_assets in assets_this_round:
                    assets_this_round_skip += 1
                else:
                    if asset_groups[:remove1] == RemoveNumber(individual_assets):
                        FP_delay_asset_capacity_dic[individual_assets] = 1
                        FP_pipeline_list_dic[asset_groups[:remove1] + '_in_pipeline'] = FP_pipeline_list_dic[
                                                                                            asset_groups[
                                                                                            :remove1] + '_in_pipeline'] - 1
                        FP_asset_capacity_dic_copy[asset_groups[:remove1]] = FP_asset_capacity_dic_copy[
                                                                                 asset_groups[:remove1]] + 1
                        delay_asset_capacity_dic_copy[asset_groups[:remove1]] = delay_asset_capacity_dic_copy[
                                                                                    asset_groups[:remove1]] + 1
                        FP_installed_counter_dic[asset_groups[:remove1] + '_installed_counter'] += 1
                        delay_installed_counter_dic[asset_groups[:remove1] + '_installed_counter'] += 1

        # Add assets that where invested in this round to merit order
        for asset_groups in investment_list_dic:
            remove1 = len(asset_groups) - len('_asset_list')
            for individual_assets in assets_this_round:
                if asset_groups[:remove1] == RemoveNumber(individual_assets):
                    FP_asset_this_round_dic[individual_assets] = 1
                    FP_pipeline_list_dic[asset_groups[:remove1] + '_in_pipeline'] = FP_pipeline_list_dic[
                                                                                        asset_groups[
                                                                                        :remove1] + '_in_pipeline'] - 1
                    FP_asset_capacity_dic_copy[asset_groups[:remove1]] = FP_asset_capacity_dic_copy[
                                                                             asset_groups[:remove1]] + 1
                    delay_asset_capacity_dic_copy[asset_groups[:remove1]] = delay_asset_capacity_dic_copy[
                                                                                asset_groups[:remove1]] + 1
                    FP_installed_counter_dic[asset_groups[:remove1] + '_installed_counter'] += 1
                    delay_installed_counter_dic[asset_groups[:remove1] + '_installed_counter'] += 1

        delay_combined_all_asset_capacity_dic = {**FP_asset_this_round_dic, **FP_delay_asset_capacity_dic,
                                                 **all_asset_capacity_dic_copy}

        for assets in investment_list:
            installed_counter_list += [str(assets) + '_installed_counter']

        FP_delay_installed_counter_dic = {key: 0 for key in installed_counter_list}

        for individual_assets in delay_combined_all_asset_capacity_dic:
            for asset_groups in investment_list:
                if RemoveNumber(individual_assets) == asset_groups:
                    FP_delay_installed_counter_dic[str(asset_groups) + '_installed_counter'] = \
                    FP_delay_installed_counter_dic[str(asset_groups) + '_installed_counter'] + 1

        # Determine for which assets there is enough budget to invest in
        FP_all_asset_capacity_dic = {}
        for asset_groups in investment_list_dic:
            remove1 = len(asset_groups) - len('_asset_list')
            if ((budget_number / energy_companies) - (
                    FP_investment_costs_dic[asset_groups[:remove1]] * equity_factor)) >= 0:
                new_asset = str(asset_groups[:remove1]) + str(
                    FP_delay_installed_counter_dic[asset_groups[:remove1] + '_installed_counter'] + 1)
                FP_all_asset_capacity_dic[new_asset] = 1
                FP_asset_capacity_dic_copy[asset_groups[:remove1]] = FP_asset_capacity_dic_copy[
                                                                         asset_groups[:remove1]] + 1
                FP_installed_counter_dic[asset_groups[:remove1] + '_installed_counter'] += 1

        FP_combined_input_all_asset_capacity_dic = cpy.deepcopy(FP_all_asset_capacity_dic)

        FP_combined_all_asset_capacity_dic = {**FP_asset_this_round_dic, **FP_combined_input_all_asset_capacity_dic,
                                              **FP_delay_asset_capacity_dic, **all_asset_capacity_dic_copy, }

        npv_dic = {}
        new_assets_list = []

        for change in FP_all_asset_capacity_dic:  # create list of the names of all the assets that are considered to be invested in
            if FP_all_asset_capacity_dic[change] > 0:
                new_assets = str(change)
                new_assets_list_input = [new_assets]
                new_assets_list += new_assets_list_input

        # Write all capacity to Linny-R for seasonal UC
        for change_asset in FP_all_asset_capacity_dic:

            change_asset_group = RemoveNumber(change_asset)

            delay_asset_capacity_dic_copy1 = cpy.deepcopy(delay_asset_capacity_dic_copy)

            delay_asset_capacity_dic_copy1[change_asset_group] = FP_asset_capacity_dic_copy[change_asset_group]

            df_onshore = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Onshore windpark output _std_weather_full_year.txt",
                names=['Onshore'])

            df_offshore = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Offshore windpark output_std_weather_full_year.txt",
                names=['Offshore'])

            df_solar = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Solar output one asset_std_weather_full_year.txt",
                names=['Solar'])

            df_UC2050_1 = pd.concat([df_onshore, df_offshore, df_solar, df_edemand, df_hdemand], axis=1)
            df_UC2050_1['Timeseries'] = 1

            capacity_dir = r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Capacity_output\_capacity.txt"

            for assets in delay_asset_capacity_dic_copy1:

                with open(
                        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Capacity_output\_capacity.txt",
                        'w') as f:
                    f.write(str(delay_asset_capacity_dic_copy1[assets]))

                capacity_exp_dir = (
                        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Capacity_output" + "\\" + str(
                    assets) + '_capacity.txt')

                copy(capacity_dir, capacity_exp_dir)

                if assets == 'OffshoreWind':
                    df_UC2050_1['Offshore'] = np.where(df_UC2050_1['Timeseries'] > 0,
                                                       df_UC2050_1['Offshore'] * (
                                                           delay_asset_capacity_dic_copy1['OffshoreWind']),
                                                       0)
                    df_UC2050_1.to_csv(
                        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Offshore windpark output_std_weather_full_year.txt",
                        columns=['Offshore'], header=None, index=None, sep=' ', mode='w')

                elif assets == 'OnshoreWind':
                    df_UC2050_1['Onshore'] = np.where(df_UC2050_1['Timeseries'] > 0,
                                                      df_UC2050_1['Onshore'] * (
                                                          delay_asset_capacity_dic_copy1['OnshoreWind']), 0)
                    df_UC2050_1.to_csv(
                        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Onshore windpark output _std_weather_full_year.txt",
                        columns=['Onshore'], header=None, index=None, sep=' ', mode='w')

                elif assets == 'SolarPV':
                    df_UC2050_1['Solar'] = np.where(df_UC2050_1['Timeseries'] > 0,
                                                    df_UC2050_1['Solar'] * (
                                                        delay_asset_capacity_dic_copy1['SolarPV']), 0)
                    df_UC2050_1.to_csv(
                        r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Solar output one asset_std_weather_full_year.txt",
                        columns=['Solar'], header=None, index=None, sep=' ', mode='w')

            df_UC2050_1.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Electricity demand_full_year k means.txt",
                columns=['E demand'], header=None, index=None, sep=' ', mode='w')
            df_UC2050_1.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Hydrogen demand_full_year k means.txt",
                columns=['H demand'], header=None, index=None, sep=' ', mode='w')
            df_UC2050_1.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Solar output one asset_std_weather_full_year.txt",
                columns=['Solar'], header=None, index=None, sep=' ', mode='w')
            df_UC2050_1.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Onshore windpark output _std_weather_full_year.txt",
                columns=['Onshore'], header=None, index=None, sep=' ', mode='w')
            df_UC2050_1.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Offshore windpark output_std_weather_full_year.txt",
                columns=['Offshore'], header=None, index=None, sep=' ', mode='w')

            # Create K-means
            df_onshore = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Onshore windpark output _std_weather_full_year.txt",
                names=['Onshore'])
            df_offshore = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Offshore windpark output_std_weather_full_year.txt",
                names=['Offshore'])
            df_solar = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Solar output one asset_std_weather_full_year.txt",
                names=['Solar'])
            df_edemand = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Electricity demand_full_year k means.txt",
                names=['E demand'])
            df_hdemand = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Hydrogen demand_full_year k means.txt",
                names=['H demand'])
            df_dd = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\dd_in_year.txt",
                names=['dd'])

            df_UC2050_2 = pd.concat([df_onshore, df_offshore, df_solar, df_edemand, df_hdemand, df_dd], axis=1)

            df_UC2050_2_header = list(df_UC2050_2)

            clusters = 24
            hours = (clusters * 24)

            df_day = df_UC2050_2.pivot_table(index=df_UC2050_2['dd'],
                                             values=df_UC2050_2_header,
                                             aggfunc='sum')

            normalized_df_day = (df_day - df_day.min()) / (df_day.max() - df_day.min())

            normalized_df_day_header = list(normalized_df_day)

            normalized_df_day = normalized_df_day[normalized_df_day_header[:6]]

            X = normalized_df_day.to_numpy()

            kmeans = KMeans(n_clusters=clusters)
            kmeans.fit(X)
            kmeans.cluster_centers_
            kmeans.labels_

            #         f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
            #         ax1.set_title('K Means')
            #         ax1.scatter(X[:,0],X[:,1],c=kmeans.labels_,cmap='rainbow')
            #         ax2.set_title("Centre")
            #         ax2.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c=range(clusters),cmap='rainbow')

            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)

            df_rep_headers = ['H demand', 'E demand', 'Onshore', 'Offshore', 'Solar']
            df_repdays = pd.DataFrame(columns=df_rep_headers)

            end_day = []

            for day in closest:
                for t in range(24):
                    end_day += [((day - 1) * 24) + t]

            for timestep in end_day:
                df_repdays = df_repdays.append(df_UC2050_2.iloc[timestep])

            df_rdsort = df_repdays
            df_rdsort.reset_index(drop=True, inplace=True)

            df_rdsort_2 = cpy.deepcopy(df_rdsort)

            df_rep_headers = ['H demand', 'E demand', 'Onshore', 'Offshore', 'Solar']

            RP_value_update_dic = {}

            for headers in df_rep_headers:
                UC_value = df_UC2050_2[headers].sum()
                RP_value = (df_rdsort[headers].sum()) * (365 / clusters)
                if (UC_value / RP_value) < 1:
                    RP_value_update = (UC_value / RP_value)
                    RP_value_update_dic[headers] = (UC_value / RP_value)
                elif (UC_value / RP_value) > 1:
                    RP_value_update = (UC_value / RP_value) * RP_value
                    RP_value_update_dic[headers] = (UC_value / RP_value)

            for value in RP_value_update_dic:
                df_rdsort_2[value] = np.where(df_rdsort_2[value] > 0,
                                              df_rdsort_2[value] * RP_value_update_dic[value], 0)

            df_rdsort_nosmooth = cpy.deepcopy(df_rdsort_2)

            df_UC2050_RD = df_rdsort_2.append(df_rdsort_2)
            df_rolling = df_UC2050_RD.rolling(24).mean()
            df_rolling.reset_index(drop=True, inplace=True)
            label = range(hours)
            df_smooth = df_rolling.drop(labels=label, axis=0)
            df_smooth.reset_index(drop=True, inplace=True)
            df_smooth.round(decimals=2)

            df_day3 = df_smooth.append(df_smooth.append(df_smooth))

            df_day3.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\E demand RP days.txt",
                columns=['E demand'], header=None, index=None, sep=' ', mode='w')
            df_day3.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\H demand RP days.txt",
                columns=['H demand'], header=None, index=None, sep=' ', mode='w')
            df_day3.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\Solar output RP days.txt",
                columns=['Solar'], header=None, index=None, sep=' ', mode='w')
            df_day3.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\Onshore output RP days.txt",
                columns=['Onshore'], header=None, index=None, sep=' ', mode='w')
            df_day3.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\Offshore output RP days.txt",
                columns=['Offshore'], header=None, index=None, sep=' ', mode='w')

            # Run seasonal UC model - RP days

            copy(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Linny-R models\sznUC2030RP.lnr",
                r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver")
            timer = 0


            while not os.path.isfile(
                    r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\sznUC2030RP-data.txt"):  #this ensures that script waits for Linny-R model to finish
                sleep(1)

            df_storage2050_UC = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\sznUC2030RP-data.txt",
                sep='\t')

            os.remove(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\sznUC2030RP-data.txt")
            os.remove(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\sznUC2030RP-log.txt")
            os.remove(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\sznUC2030RP-stats.txt")

            # Import seasonal t0 storage targets
            t0_HSE = (df_storage2050_UC.loc[576, 'Hydrogen Storage - Electrolyser S|L']) * (365 / clusters)
            t0_HSCCGT = (df_storage2050_UC.loc[576, 'Hydrogen Storage - Electrolyser CCGT|L']) * (365 / clusters)
            t0_HSOCGT = (df_storage2050_UC.loc[576, 'Hydrogen Storage - Electrolyser OCGT|L']) * (365 / clusters)

            # Import only second year
            label1 = range(0, 577)
            df_storage_y2_UC = df_storage2050_UC.drop(labels=label1, axis=0)

            label2 = range(1153, 1441)
            df_storage_y2_UC = df_storage_y2_UC.drop(labels=label2, axis=0)

            df_storage_y2_UC.reset_index(drop=True, inplace=True)
            df_storage_y2_UC.index = np.arange(1, len(df_storage_y2_UC) + 1)

            df_begin_day = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Seas storage txt files\begin_of_day.txt",
                names=['Begin day'])
            df_end_day = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Seas storage txt files\end_of_day.txt",
                names=['End of day'])
            df_dd = pd.read_csv(r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Seas storage txt files\dd.txt",
                                names=['dd'])
            df_begin_day.index = np.arange(1, len(df_begin_day) + 1)
            df_end_day.index = np.arange(1, len(df_end_day) + 1)
            df_dd.index = np.arange(1, len(df_dd) + 1)

            df_rep_headers = ['LB Storage', 'UB Storage']
            df_storage_comp_headers = ['Charge', 'Discharge', 'Target']

            df_storagetarget_UC_HSE = pd.DataFrame()
            df_storagetarget_UC_HSCCGT = pd.DataFrame()
            df_storagetarget_UC_HSOCGT = pd.DataFrame()

            df_storage_y2_UC = pd.concat([df_storage_y2_UC, df_begin_day, df_end_day, df_dd], axis=1)

            df_storagetarget_UC_HSE['Begin day targets'] = np.where(df_storage_y2_UC['Begin day'] > 0, df_storage_y2_UC[
                'Hydrogen Storage - Electrolyser S|L'], 0)
            df_storagetarget_UC_HSE['End day targets'] = np.where(df_storage_y2_UC['End of day'] > 0, df_storage_y2_UC[
                'Hydrogen Storage - Electrolyser S|L'], 0)
            df_storagetarget_UC_HSE.index = np.arange(1, len(df_storagetarget_UC_HSE) + 1)
            df_storagetarget_UC_HSE = pd.concat([df_storagetarget_UC_HSE, df_dd], axis=1)

            df_storagetarget_UC_HSCCGT['Begin day targets'] = np.where(df_storage_y2_UC['Begin day'] > 0,
                                                                       df_storage_y2_UC[
                                                                           'Hydrogen Storage - Electrolyser CCGT|L'], 0)
            df_storagetarget_UC_HSCCGT['End day targets'] = np.where(df_storage_y2_UC['End of day'] > 0,
                                                                     df_storage_y2_UC[
                                                                         'Hydrogen Storage - Electrolyser CCGT|L'], 0)
            df_storagetarget_UC_HSCCGT.index = np.arange(1, len(df_storagetarget_UC_HSCCGT) + 1)
            df_storagetarget_UC_HSCCGT = pd.concat([df_storagetarget_UC_HSCCGT, df_dd], axis=1)

            df_storagetarget_UC_HSOCGT['Begin day targets'] = np.where(df_storage_y2_UC['Begin day'] > 0,
                                                                       df_storage_y2_UC[
                                                                           'Hydrogen Storage - Electrolyser OCGT|L'], 0)
            df_storagetarget_UC_HSOCGT['End day targets'] = np.where(df_storage_y2_UC['End of day'] > 0,
                                                                     df_storage_y2_UC[
                                                                         'Hydrogen Storage - Electrolyser OCGT|L'], 0)
            df_storagetarget_UC_HSOCGT.index = np.arange(1, len(df_storagetarget_UC_HSOCGT) + 1)
            df_storagetarget_UC_HSOCGT = pd.concat([df_storagetarget_UC_HSOCGT, df_dd], axis=1)

            df_storagetarget_UC_HSE_dd = df_storagetarget_UC_HSE.pivot_table(index=df_storagetarget_UC_HSE['dd'],
                                                                             values=list(df_storagetarget_UC_HSE),
                                                                             aggfunc='sum')

            df_storagetarget_UC_HSCCGT_dd = df_storagetarget_UC_HSCCGT.pivot_table(
                index=df_storagetarget_UC_HSCCGT['dd'],
                values=list(df_storagetarget_UC_HSCCGT),
                aggfunc='sum')

            df_storagetarget_UC_HSOCGT_dd = df_storagetarget_UC_HSOCGT.pivot_table(
                index=df_storagetarget_UC_HSOCGT['dd'],
                values=list(df_storagetarget_UC_HSOCGT),
                aggfunc='sum')

            for index, colum in df_storagetarget_UC_HSE_dd.iterrows():
                if index == 1:
                    df_storagetarget_UC_HSE_dd.loc[index, 'UB'] = t0_HSE + ((df_storagetarget_UC_HSE_dd.loc[
                                                                                 index, 'End day targets'] -
                                                                             df_storagetarget_UC_HSE_dd.loc[
                                                                                 index, 'Begin day targets']) * (
                                                                                        365 / clusters))
                    df_storagetarget_UC_HSE_dd.loc[index, 'Compensation'] = (df_storagetarget_UC_HSE_dd.loc[
                                                                                 index, 'End day targets'] -
                                                                             df_storagetarget_UC_HSE_dd.loc[
                                                                                 index, 'Begin day targets']) * (
                                                                                        (365 / clusters) - 1)
                    cluster_reducer = 0
                    df_storagetarget_UC_HSE_dd.loc[index, 'Error'] = df_storagetarget_UC_HSE_dd.loc[index, 'UB'] + \
                                                                     df_storagetarget_UC_HSE_dd.loc[
                                                                         index, 'Compensation']
                    while df_storagetarget_UC_HSE_dd.loc[index, 'Error'] < 0:
                        cluster_reducer += 1
                        df_storagetarget_UC_HSE_dd.loc[index, 'Compensation'] = (df_storagetarget_UC_HSE_dd.loc[
                                                                                     index, 'End day targets'] -
                                                                                 df_storagetarget_UC_HSE_dd.loc[
                                                                                     index, 'Begin day targets']) * ((
                                                                                                                                 365 / clusters) - 1 - cluster_reducer)
                        df_storagetarget_UC_HSE_dd.loc[index, 'Error'] = df_storagetarget_UC_HSE_dd.loc[index, 'UB'] + \
                                                                         df_storagetarget_UC_HSE_dd.loc[
                                                                             index, 'Compensation']
                        if cluster_reducer == clusters:
                            df_storagetarget_UC_HSE_dd.loc[index, 'Error'] = 0

                if index > 1:
                    df_storagetarget_UC_HSE_dd.loc[index, 'UB'] = df_storagetarget_UC_HSE_dd.loc[index - 1, 'UB'] + ((
                                                                                                                                 df_storagetarget_UC_HSE_dd.loc[
                                                                                                                                     index, 'End day targets'] -
                                                                                                                                 df_storagetarget_UC_HSE_dd.loc[
                                                                                                                                     index, 'Begin day targets']) * (
                                                                                                                                 365 / clusters))
                    df_storagetarget_UC_HSE_dd.loc[index, 'Compensation'] = (df_storagetarget_UC_HSE_dd.loc[
                                                                                 index, 'End day targets'] -
                                                                             df_storagetarget_UC_HSE_dd.loc[
                                                                                 index, 'Begin day targets']) * (
                                                                                        (365 / clusters) - 1)
                    cluster_reducer = 0
                    df_storagetarget_UC_HSE_dd.loc[index, 'Error'] = df_storagetarget_UC_HSE_dd.loc[index, 'UB'] + \
                                                                     df_storagetarget_UC_HSE_dd.loc[
                                                                         index, 'Compensation']
                    while df_storagetarget_UC_HSE_dd.loc[index, 'Error'] < 0:
                        cluster_reducer += 1
                        df_storagetarget_UC_HSE_dd.loc[index, 'Compensation'] = (df_storagetarget_UC_HSE_dd.loc[
                                                                                     index, 'End day targets'] -
                                                                                 df_storagetarget_UC_HSE_dd.loc[
                                                                                     index, 'Begin day targets']) * ((
                                                                                                                                 365 / clusters) - 1 - cluster_reducer)
                        df_storagetarget_UC_HSE_dd.loc[index, 'Error'] = df_storagetarget_UC_HSE_dd.loc[index, 'UB'] + \
                                                                         df_storagetarget_UC_HSE_dd.loc[
                                                                             index, 'Compensation']
                        if cluster_reducer == clusters:
                            df_storagetarget_UC_HSE_dd.loc[index, 'Error'] = 0

            for index, colum in df_storagetarget_UC_HSCCGT_dd.iterrows():
                if index == 1:
                    df_storagetarget_UC_HSCCGT_dd.loc[index, 'UB'] = t0_HSCCGT + ((df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                       index, 'End day targets'] -
                                                                                   df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                       index, 'Begin day targets']) * (
                                                                                              365 / clusters))
                    df_storagetarget_UC_HSCCGT_dd.loc[index, 'Compensation'] = (df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                    index, 'End day targets'] -
                                                                                df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                    index, 'Begin day targets']) * (
                                                                                           (365 / clusters) - 1)
                    cluster_reducer = 0
                    df_storagetarget_UC_HSCCGT_dd.loc[index, 'Error'] = df_storagetarget_UC_HSCCGT_dd.loc[index, 'UB'] + \
                                                                        df_storagetarget_UC_HSCCGT_dd.loc[
                                                                            index, 'Compensation']
                    while df_storagetarget_UC_HSCCGT_dd.loc[index, 'Error'] < 0:
                        cluster_reducer += 1
                        df_storagetarget_UC_HSCCGT_dd.loc[index, 'Compensation'] = (df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                        index, 'End day targets'] -
                                                                                    df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                        index, 'Begin day targets']) * (
                                                                                               (
                                                                                                           365 / clusters) - 1 - cluster_reducer)
                        df_storagetarget_UC_HSCCGT_dd.loc[index, 'Error'] = df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                index, 'UB'] + \
                                                                            df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                index, 'Compensation']
                        if cluster_reducer == clusters:
                            df_storagetarget_UC_HSCCGT_dd.loc[index, 'Error'] = 0

                if index > 1:
                    df_storagetarget_UC_HSCCGT_dd.loc[index, 'UB'] = df_storagetarget_UC_HSCCGT_dd.loc[
                                                                         index - 1, 'UB'] + ((
                                                                                                         df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                                             index, 'End day targets'] -
                                                                                                         df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                                             index, 'Begin day targets']) * (
                                                                                                         365 / clusters))
                    df_storagetarget_UC_HSCCGT_dd.loc[index, 'Compensation'] = (df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                    index, 'End day targets'] -
                                                                                df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                    index, 'Begin day targets']) * (
                                                                                           (365 / clusters) - 1)
                    cluster_reducer = 0
                    df_storagetarget_UC_HSCCGT_dd.loc[index, 'Error'] = df_storagetarget_UC_HSCCGT_dd.loc[index, 'UB'] + \
                                                                        df_storagetarget_UC_HSCCGT_dd.loc[
                                                                            index, 'Compensation']
                    while df_storagetarget_UC_HSCCGT_dd.loc[index, 'Error'] < 0:
                        cluster_reducer += 1
                        df_storagetarget_UC_HSCCGT_dd.loc[index, 'Compensation'] = (df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                        index, 'End day targets'] -
                                                                                    df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                        index, 'Begin day targets']) * (
                                                                                               (
                                                                                                           365 / clusters) - 1 - cluster_reducer)
                        df_storagetarget_UC_HSCCGT_dd.loc[index, 'Error'] = df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                index, 'UB'] + \
                                                                            df_storagetarget_UC_HSCCGT_dd.loc[
                                                                                index, 'Compensation']
                        if cluster_reducer == clusters:
                            df_storagetarget_UC_HSCCGT_dd.loc[index, 'Error'] = 0

            for index, colum in df_storagetarget_UC_HSOCGT_dd.iterrows():
                if index == 1:
                    df_storagetarget_UC_HSOCGT_dd.loc[index, 'UB'] = t0_HSOCGT + ((df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                       index, 'End day targets'] -
                                                                                   df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                       index, 'Begin day targets']) * (
                                                                                              365 / clusters))
                    df_storagetarget_UC_HSOCGT_dd.loc[index, 'Compensation'] = (df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                    index, 'End day targets'] -
                                                                                df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                    index, 'Begin day targets']) * (
                                                                                           (365 / clusters) - 1)
                    cluster_reducer = 0
                    df_storagetarget_UC_HSOCGT_dd.loc[index, 'Error'] = df_storagetarget_UC_HSOCGT_dd.loc[index, 'UB'] + \
                                                                        df_storagetarget_UC_HSOCGT_dd.loc[
                                                                            index, 'Compensation']
                    while df_storagetarget_UC_HSOCGT_dd.loc[index, 'Error'] < 0:
                        cluster_reducer += 1
                        df_storagetarget_UC_HSOCGT_dd.loc[index, 'Compensation'] = (df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                        index, 'End day targets'] -
                                                                                    df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                        index, 'Begin day targets']) * (
                                                                                               (
                                                                                                           365 / clusters) - 1 - cluster_reducer)
                        df_storagetarget_UC_HSOCGT_dd.loc[index, 'Error'] = df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                index, 'UB'] + \
                                                                            df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                index, 'Compensation']
                        if cluster_reducer == clusters:
                            df_storagetarget_UC_HSOCGT_dd.loc[index, 'Error'] = 0

                if index > 1:
                    df_storagetarget_UC_HSOCGT_dd.loc[index, 'UB'] = df_storagetarget_UC_HSOCGT_dd.loc[
                                                                         index - 1, 'UB'] + ((
                                                                                                         df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                                             index, 'End day targets'] -
                                                                                                         df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                                             index, 'Begin day targets']) * (
                                                                                                         365 / clusters))
                    df_storagetarget_UC_HSOCGT_dd.loc[index, 'Compensation'] = (df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                    index, 'End day targets'] -
                                                                                df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                    index, 'Begin day targets']) * (
                                                                                           (365 / clusters) - 1)
                    cluster_reducer = 0
                    df_storagetarget_UC_HSOCGT_dd.loc[index, 'Error'] = df_storagetarget_UC_HSOCGT_dd.loc[index, 'UB'] + \
                                                                        df_storagetarget_UC_HSOCGT_dd.loc[
                                                                            index, 'Compensation']
                    while df_storagetarget_UC_HSOCGT_dd.loc[index, 'Error'] < 0:
                        cluster_reducer += 1
                        df_storagetarget_UC_HSOCGT_dd.loc[index, 'Compensation'] = (df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                        index, 'End day targets'] -
                                                                                    df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                        index, 'Begin day targets']) * (
                                                                                               (
                                                                                                           365 / clusters) - 1 - cluster_reducer)
                        df_storagetarget_UC_HSOCGT_dd.loc[index, 'Error'] = df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                index, 'UB'] + \
                                                                            df_storagetarget_UC_HSOCGT_dd.loc[
                                                                                index, 'Compensation']
                        if cluster_reducer == clusters:
                            df_storagetarget_UC_HSOCGT_dd.loc[index, 'Error'] = 0

            for index, column in df_storagetarget_UC_HSE.iterrows():
                if (index / 24).is_integer():
                    df_storagetarget_UC_HSE.loc[index, 'UB'] = df_storagetarget_UC_HSE_dd.loc[index / 24, 'UB']
                    df_storagetarget_UC_HSE.loc[index, 'LB'] = df_storagetarget_UC_HSE_dd.loc[index / 24, 'UB']

                    if df_storagetarget_UC_HSE_dd.loc[index / 24, 'Compensation'] > 0:
                        df_storagetarget_UC_HSE.loc[index, 'Charge'] = df_storagetarget_UC_HSE_dd.loc[
                            index / 24, 'Compensation']
                        df_storagetarget_UC_HSE.loc[index, 'Discharge'] = 0
                    else:
                        df_storagetarget_UC_HSE.loc[index, 'Discharge'] = df_storagetarget_UC_HSE_dd.loc[
                            index / 24, 'Compensation']
                        df_storagetarget_UC_HSE.loc[index, 'Charge'] = 0
                else:
                    df_storagetarget_UC_HSE.loc[index, 'Charge'] = 0
                    df_storagetarget_UC_HSE.loc[index, 'Discharge'] = 0
                    df_storagetarget_UC_HSE.loc[index, 'LB'] = 0
                    df_storagetarget_UC_HSE.loc[index, 'UB'] = 1e9

            for index, column in df_storagetarget_UC_HSCCGT.iterrows():
                if (index / 24).is_integer():
                    df_storagetarget_UC_HSCCGT.loc[index, 'UB'] = df_storagetarget_UC_HSCCGT_dd.loc[index / 24, 'UB']
                    df_storagetarget_UC_HSCCGT.loc[index, 'LB'] = df_storagetarget_UC_HSCCGT_dd.loc[index / 24, 'UB']

                    if df_storagetarget_UC_HSCCGT_dd.loc[index / 24, 'Compensation'] > 0:
                        df_storagetarget_UC_HSCCGT.loc[index, 'Charge'] = df_storagetarget_UC_HSCCGT_dd.loc[
                            index / 24, 'Compensation']
                        df_storagetarget_UC_HSCCGT.loc[index, 'Discharge'] = 0
                    else:
                        df_storagetarget_UC_HSCCGT.loc[index, 'Discharge'] = df_storagetarget_UC_HSCCGT_dd.loc[
                            index / 24, 'Compensation']
                        df_storagetarget_UC_HSCCGT.loc[index, 'Charge'] = 0
                else:
                    df_storagetarget_UC_HSCCGT.loc[index, 'Charge'] = 0
                    df_storagetarget_UC_HSCCGT.loc[index, 'Discharge'] = 0
                    df_storagetarget_UC_HSCCGT.loc[index, 'LB'] = 0
                    df_storagetarget_UC_HSCCGT.loc[index, 'UB'] = 1e9

            for index, column in df_storagetarget_UC_HSOCGT.iterrows():
                if (index / 24).is_integer():
                    df_storagetarget_UC_HSOCGT.loc[index, 'UB'] = df_storagetarget_UC_HSOCGT_dd.loc[index / 24, 'UB']
                    df_storagetarget_UC_HSOCGT.loc[index, 'LB'] = df_storagetarget_UC_HSOCGT_dd.loc[index / 24, 'UB']

                    if df_storagetarget_UC_HSOCGT_dd.loc[index / 24, 'Compensation'] > 0:
                        df_storagetarget_UC_HSOCGT.loc[index, 'Charge'] = df_storagetarget_UC_HSOCGT_dd.loc[
                            index / 24, 'Compensation']
                        df_storagetarget_UC_HSOCGT.loc[index, 'Discharge'] = 0
                    else:
                        df_storagetarget_UC_HSOCGT.loc[index, 'Discharge'] = df_storagetarget_UC_HSOCGT_dd.loc[
                            index / 24, 'Compensation']
                        df_storagetarget_UC_HSOCGT.loc[index, 'Charge'] = 0
                else:
                    df_storagetarget_UC_HSOCGT.loc[index, 'Charge'] = 0
                    df_storagetarget_UC_HSOCGT.loc[index, 'Discharge'] = 0
                    df_storagetarget_UC_HSOCGT.loc[index, 'LB'] = 0
                    df_storagetarget_UC_HSOCGT.loc[index, 'UB'] = 1e9

            df_storagetarget_UC_HSE['T0 time'] = 1
            df_storagetarget_UC_HSE['t0_HSE'] = np.where(df_storagetarget_UC_HSE['T0 time'] >= 0, 0, 0)
            df_storagetarget_UC_HSE.loc[1, 't0_HSE'] = t0_HSE

            df_storagetarget_UC_HSCCGT['T0 time'] = 1
            df_storagetarget_UC_HSCCGT['t0_HSCCGT'] = np.where(df_storagetarget_UC_HSCCGT['T0 time'] >= 0, 0, 0)
            df_storagetarget_UC_HSCCGT.loc[1, 't0_HSCCGT'] = t0_HSCCGT

            df_storagetarget_UC_HSOCGT['T0 time'] = 1
            df_storagetarget_UC_HSOCGT['t0_HSOCGT'] = np.where(df_storagetarget_UC_HSOCGT['T0 time'] >= 0, 0, 0)
            df_storagetarget_UC_HSOCGT.loc[1, 't0_HSOCGT'] = t0_HSOCGT

            # Export storage targets to Linny-R text files
            df_storagetarget_UC_HSE.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\LB_HSE.txt",
                columns=['LB'], header=None, index=None, sep=' ', mode='w')
            df_storagetarget_UC_HSE.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\UB_HSE.txt",
                columns=['UB'], header=None, index=None, sep=' ', mode='w')
            df_storagetarget_UC_HSE.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\t0_target_HSE.txt",
                columns=['t0_HSE'], header=None, index=None, sep=' ', mode='w')
            df_storagetarget_UC_HSE.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\HSE_in.txt",
                columns=['Charge'], header=None, index=None, sep=' ', mode='w')
            df_storagetarget_UC_HSE.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\HSE_out.txt",
                columns=['Discharge'], header=None, index=None, sep=' ', mode='w')

            df_storagetarget_UC_HSCCGT.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\LB_HSCCGT.txt",
                columns=['LB'], header=None, index=None, sep=' ', mode='w')
            df_storagetarget_UC_HSCCGT.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\UB_HSCCGT.txt",
                columns=['UB'], header=None, index=None, sep=' ', mode='w')
            df_storagetarget_UC_HSOCGT.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\LB_HSOCGT.txt",
                columns=['LB'], header=None, index=None, sep=' ', mode='w')
            df_storagetarget_UC_HSOCGT.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\UB_HSOCGT.txt",
                columns=['UB'], header=None, index=None, sep=' ', mode='w')

            df_storagetarget_UC_HSCCGT.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\t0_target_HSCCGT.txt",
                columns=['t0_HSCCGT'], header=None, index=None, sep=' ', mode='w')
            df_storagetarget_UC_HSOCGT.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\t0_target_HSOCGT.txt",
                columns=['t0_HSOCGT'], header=None, index=None, sep=' ', mode='w')

            df_storagetarget_UC_HSCCGT.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\HSCCGT_in.txt",
                columns=['Charge'], header=None, index=None, sep=' ', mode='w')
            df_storagetarget_UC_HSCCGT.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\HSCCGT_out.txt",
                columns=['Discharge'], header=None, index=None, sep=' ', mode='w')
            df_storagetarget_UC_HSOCGT.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\HSOCGT_in.txt",
                columns=['Charge'], header=None, index=None, sep=' ', mode='w')
            df_storagetarget_UC_HSOCGT.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\HSOCGT_out.txt",
                columns=['Discharge'], header=None, index=None, sep=' ', mode='w')

            # Change input to not smoothened timeseries
            df_rdsort_nosmooth.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\E demand RP days.txt",
                columns=['E demand'], header=None, index=None, sep=' ', mode='w')
            df_rdsort_nosmooth.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\H demand RP days.txt",
                columns=['H demand'], header=None, index=None, sep=' ', mode='w')
            df_rdsort_nosmooth.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\Solar output RP days.txt",
                columns=['Solar'], header=None, index=None, sep=' ', mode='w')
            df_rdsort_nosmooth.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\Onshore output RP days.txt",
                columns=['Onshore'], header=None, index=None, sep=' ', mode='w')
            df_rdsort_nosmooth.to_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\Offshore output RP days.txt",
                columns=['Offshore'], header=None, index=None, sep=' ', mode='w')

            # Run daily UC model - RP days
            copy(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Linny-R models\dailyUC2030RP.lnr",
                r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver")
            timer = 0


            while not os.path.isfile(
                    r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\dailyUC2030RP-data.txt"):  #this ensures script waits on Linny-R model to finish
                sleep(1)
                timer += 1

            df_linny = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\dailyUC2030RP-data.txt",
                sep='\t')

            df_linny.drop(index=df_linny.index[0], axis=0, inplace=True)
            df_linny.reset_index(drop=True, inplace=True)

            os.remove(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\dailyUC2030RP-data.txt")
            os.remove(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\dailyUC2030RP-log.txt")
            os.remove(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\2030 receiver\dailyUC2030RP-stats.txt")

            df_linny.columns = df_linny.columns.str.replace(' ', '', regex=True)
            df_linny.columns = df_linny.columns.str.replace('|', '', regex=True)

            # Calculate REV

            # Update max output of capacities to match rep days
            df_offshore = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\Offshore output RP days.txt",
                names=['Offshore'])
            df_onshore = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\Onshore output RP days.txt",
                names=['Onshore'])
            df_solar = pd.read_csv(
                r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries rep day\Solar output RP days.txt",
                names=['Solar'])

            df_onshore['Time'] = 1
            df_offshore['Time'] = 1
            df_solar['Time'] = 1

            df_onshore['Onshore'] = np.where(df_onshore['Time'] == 1,
                                             df_onshore['Onshore'] / delay_asset_capacity_dic_copy1['OnshoreWind'],
                                             df_onshore['Onshore'])
            df_offshore['Offshore'] = np.where(df_offshore['Time'] == 1,
                                               df_offshore['Offshore'] / delay_asset_capacity_dic_copy1[
                                                   'OffshoreWind'],
                                               df_offshore['Offshore'])
            df_solar['Solar'] = np.where(df_solar['Time'] == 1,
                                         df_solar['Solar'] / delay_asset_capacity_dic_copy1['SolarPV'],
                                         df_solar['Solar'])

            # Update e-price

            asset_MC_dic_sorted = sorted(asset_MC_dic.items(), key=lambda x: x[1], reverse=True)

            counter = 0

            asset_MC_dic_sorted = sorted(asset_MC_dic.items(), key=lambda x: x[1], reverse=True)

            for index, column in df_linny.iterrows():
                for assets in asset_MC_dic_sorted:
                    if investment_list_classifier_dic[assets[0]] != 'Storage':
                        if df_linny.loc[index, asset_mc_linny_output_dic[assets[0]]] > 0:
                            E_price_during_t = assets[1]
                            df_linny.loc[index, 'e-price'] = E_price_during_t
                            counter += 1
                            break

            df_linny['e-price'] = np.where(df_linny['VOLLL'] > 0, VOLL, df_linny['e-price'])
            df_linny['e-price'] = np.where(df_linny['HydrogenimportL'] > 0, VOLL, df_linny['e-price'])

            df_tot_data_UC = pd.concat([df_linny, df_offshore, df_onshore, df_solar], axis=1)

            investment_list_dic_FP = {}

            for assets in investment_list:
                investment_list_list += [str(assets) + '_asset_list']

            investment_list_dic_FP = {key: [] for key in investment_list_list}

            for individual_assets in delay_combined_all_asset_capacity_dic:
                if delay_combined_all_asset_capacity_dic[individual_assets] > 0:
                    investment_list_dic_FP[str(RemoveNumber(individual_assets)) + '_asset_list'] += [individual_assets]

            investment_list_dic_FP[str(RemoveNumber(change_asset)) + '_asset_list'] = [change_asset] + \
                                                                                      investment_list_dic_FP[
                                                                                          str(RemoveNumber(
                                                                                              change_asset)) + '_asset_list']
            for individual_assets in FP_asset_this_round_dic:
                this_round_list = investment_list_dic_FP[RemoveNumber(individual_assets) + '_asset_list']
                this_round_list.remove(individual_assets)
                investment_list_dic_FP[RemoveNumber(individual_assets) + '_asset_list'] = [individual_assets] + \
                                                                                          investment_list_dic_FP[
                                                                                              RemoveNumber(
                                                                                                  individual_assets) + '_asset_list']


            for asset_groups in investment_list_dic_FP:
                remove1 = len(asset_groups) - len('_asset_list')
                if investment_list_classifier_dic[asset_groups[:remove1]] == 'Asset':
                    max_output_asset = standard_capacity_dic[asset_groups[:remove1]]
                    for individual_assets in investment_list_dic_FP[asset_groups]:
                        if FP_combined_all_asset_capacity_dic[individual_assets] == 1:
                            df_tot_data_UC[individual_assets + '_output'] = np.where(
                                df_tot_data_UC[
                                    asset_linny_output_dic[asset_groups[:remove1]]] - max_output_asset > 0,
                                max_output_asset, df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]])

                            df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]] = np.where(
                                df_tot_data_UC[
                                    asset_linny_output_dic[asset_groups[:remove1]]] - max_output_asset > 0,
                                df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]] - max_output_asset,
                                0)

                            df_tot_data_UC[individual_assets] = np.where(
                                df_tot_data_UC[individual_assets + '_output'] > 0,
                                df_tot_data_UC[individual_assets + '_output'] * (
                                        df_tot_data_UC['e-price'] - asset_MC_dic[
                                    asset_groups[:remove1]]), 0)

                if investment_list_classifier_dic[asset_groups[:remove1]] == 'VRES':
                    for individual_assets in investment_list_dic_FP[asset_groups]:

                        if FP_combined_all_asset_capacity_dic[individual_assets] == 1:
                            df_tot_data_UC[individual_assets + '_output'] = np.where(
                                df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]] - df_tot_data_UC[
                                    weather_dic[asset_groups[:remove1]]] > 0,
                                df_tot_data_UC[weather_dic[asset_groups[:remove1]]],
                                df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]])
                            df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]] = np.where(
                                df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]] - df_tot_data_UC[
                                    weather_dic[asset_groups[:remove1]]] > 0,
                                df_tot_data_UC[asset_linny_output_dic[asset_groups[:remove1]]] - df_tot_data_UC[
                                    weather_dic[asset_groups[:remove1]]], 0)
                            df_tot_data_UC[individual_assets] = np.where(
                                df_tot_data_UC[individual_assets + '_output'] > 0,
                                df_tot_data_UC[individual_assets + '_output'] *
                                df_tot_data_UC['e-price'], 0)

                if investment_list_classifier_dic[asset_groups[:remove1]] == 'Storage':

                    if storage_list_classifier_dic[asset_groups[:remove1]] == 'Hydrogen':

                        for individual_assets in investment_list_dic_FP[asset_groups]:

                            if FP_combined_all_asset_capacity_dic[individual_assets] == 1:

                                if charge_storage_dic[
                                    investment_storage_dic[asset_groups[:remove1]][0]] == 'Charge':
                                    charge_asset = investment_storage_dic[asset_groups[:remove1]][0]
                                    max_output_charge_asset = standard_capacity_dic[charge_asset]

                                    df_tot_data_UC[individual_assets + '_Charge'] = np.where(
                                        df_tot_data_UC[
                                            asset_linny_output_dic[charge_asset]] - max_output_charge_asset > 0,
                                        max_output_charge_asset,
                                        df_tot_data_UC[asset_linny_output_dic[charge_asset]])
                                    df_tot_data_UC[asset_linny_output_dic[charge_asset]] = np.where(
                                        df_tot_data_UC[
                                            asset_linny_output_dic[charge_asset]] - max_output_charge_asset > 0,
                                        df_tot_data_UC[
                                            asset_linny_output_dic[charge_asset]] - max_output_charge_asset, 0)
                                    df_tot_data_UC[individual_assets + '_ChargeRev'] = np.where(
                                        df_tot_data_UC[individual_assets + '_Charge'] > 0,
                                        df_tot_data_UC[individual_assets + '_Charge'] * (
                                                df_tot_data_UC['e-price'] - 3), 0)

                                if discharge_storage_dic[
                                    investment_storage_dic[asset_groups[:remove1]][1]] != 'ElectrolyserSDischarge':

                                    discharge_asset = investment_storage_dic[asset_groups[:remove1]][1]
                                    max_output_discharge_asset = standard_capacity_dic[discharge_asset]

                                    if discharge_storage_dic[
                                        investment_storage_dic[asset_groups[:remove1]][1]] == 'Discharge':
                                        df_tot_data_UC[individual_assets + '_Discharge'] = np.where(df_tot_data_UC[
                                                                                                        asset_linny_output_dic[
                                                                                                            discharge_asset]] - max_output_discharge_asset > 0,
                                                                                                    max_output_discharge_asset,
                                                                                                    df_tot_data_UC[
                                                                                                        asset_linny_output_dic[
                                                                                                            discharge_asset]])
                                        df_tot_data_UC[asset_linny_output_dic[discharge_asset]] = np.where(
                                            df_tot_data_UC[
                                                asset_linny_output_dic[
                                                    discharge_asset]] - max_output_discharge_asset > 0,
                                            df_tot_data_UC[
                                                asset_linny_output_dic[
                                                    discharge_asset]] - max_output_discharge_asset,
                                            0)
                                        df_tot_data_UC[individual_assets + '_DischargeRev'] = np.where(
                                            df_tot_data_UC[individual_assets + '_Discharge'] > 0,
                                            df_tot_data_UC[individual_assets + '_Discharge'] * (
                                                df_tot_data_UC['e-price']), 0)

                                else:
                                    discharge_asset = investment_storage_dic[asset_groups[:remove1]][1]
                                    max_output_discharge_asset = standard_capacity_dic[discharge_asset]

                                    if discharge_storage_dic[
                                        investment_storage_dic[asset_groups[:remove1]][1]] == 'Discharge':
                                        df_tot_data_UC[individual_assets + '_Discharge'] = np.where(
                                            df_tot_data_UC[asset_linny_output_dic[discharge_asset]] > 0,
                                            df_tot_data_UC[asset_linny_output_dic[discharge_asset]], 0)
                                        df_tot_data_UC[individual_assets + '_DischargeRev'] = np.where(
                                            df_tot_data_UC[individual_assets + '_Discharge'] > 0,
                                            df_tot_data_UC[individual_assets + '_Discharge'] * (
                                                df_tot_data_UC['e-price']), 0)
                    else:
                        for individual_assets in investment_list_dic_FP[asset_groups]:

                            if FP_combined_all_asset_capacity_dic[individual_assets] == 1:

                                if charge_storage_dic[
                                    investment_storage_dic[asset_groups[:remove1]][0]] == 'Charge':
                                    charge_asset = investment_storage_dic[asset_groups[:remove1]][0]
                                    max_output_charge_asset = standard_capacity_dic[charge_asset]

                                    df_tot_data_UC[individual_assets + '_Charge'] = np.where(
                                        df_tot_data_UC[
                                            asset_linny_output_dic[charge_asset]] - max_output_charge_asset > 0,
                                        max_output_charge_asset,
                                        df_tot_data_UC[asset_linny_output_dic[charge_asset]])
                                    df_tot_data_UC[asset_linny_output_dic[charge_asset]] = np.where(
                                        df_tot_data_UC[
                                            asset_linny_output_dic[charge_asset]] - max_output_charge_asset > 0,
                                        df_tot_data_UC[
                                            asset_linny_output_dic[charge_asset]] - max_output_charge_asset, 0)
                                    df_tot_data_UC[individual_assets + '_ChargeRev'] = np.where(
                                        df_tot_data_UC[individual_assets + '_Charge'] > 0,
                                        df_tot_data_UC[individual_assets + '_Charge'] * df_tot_data_UC['e-price'],
                                        0)

                                if discharge_storage_dic[
                                    investment_storage_dic[asset_groups[:remove1]][1]] != 'ElectrolyserSDischarge':

                                    discharge_asset = investment_storage_dic[asset_groups[:remove1]][1]
                                    max_output_discharge_asset = standard_capacity_dic[discharge_asset]

                                    if discharge_storage_dic[
                                        investment_storage_dic[asset_groups[:remove1]][1]] == 'Discharge':
                                        df_tot_data_UC[individual_assets + '_Discharge'] = np.where(df_tot_data_UC[
                                                                                                        asset_linny_output_dic[
                                                                                                            discharge_asset]] - max_output_discharge_asset > 0,
                                                                                                    max_output_discharge_asset,
                                                                                                    df_tot_data_UC[
                                                                                                        asset_linny_output_dic[
                                                                                                            discharge_asset]])
                                        df_tot_data_UC[asset_linny_output_dic[discharge_asset]] = np.where(
                                            df_tot_data_UC[
                                                asset_linny_output_dic[
                                                    discharge_asset]] - max_output_discharge_asset > 0,
                                            df_tot_data_UC[
                                                asset_linny_output_dic[
                                                    discharge_asset]] - max_output_discharge_asset,
                                            0)
                                        df_tot_data_UC[individual_assets + '_DischargeRev'] = np.where(
                                            df_tot_data_UC[individual_assets + '_Discharge'] > 0,
                                            df_tot_data_UC[individual_assets + '_Discharge'] * df_tot_data_UC[
                                                'e-price'], 0)

                                else:
                                    discharge_asset = investment_storage_dic[asset_groups[:remove1]][1]
                                    max_output_discharge_asset = standard_capacity_dic[discharge_asset]

                                    if discharge_storage_dic[
                                        investment_storage_dic[asset_groups[:remove1]][1]] == 'Discharge':
                                        df_tot_data_UC[individual_assets + '_Discharge'] = np.where(
                                            df_tot_data_UC[asset_linny_output_dic[discharge_asset]] > 0,
                                            df_tot_data_UC[asset_linny_output_dic[discharge_asset]], 0)
                                        df_tot_data_UC[individual_assets + '_DischargeRev'] = np.where(
                                            df_tot_data_UC[individual_assets + '_Discharge'] > 0,
                                            df_tot_data_UC[individual_assets + '_Discharge'] * df_tot_data_UC[
                                                'e-price'], 0)

            df_asset_cf_UC = pd.DataFrame()

            for asset_groups in investment_list_dic_FP:

                remove1 = len(asset_groups) - len('_asset_list')

                if investment_list_classifier_dic[asset_groups[:remove1]] == 'Asset':
                    for individual_assets in investment_list_dic_FP[asset_groups]:
                        if FP_combined_all_asset_capacity_dic[individual_assets] == 1:
                            CF = (df_tot_data_UC[individual_assets].sum() * (365 / clusters)) - FP_asset_OM_dic[
                                asset_groups[:remove1]]
                            df_asset_cf_UC.loc[year, individual_assets] = CF

                elif investment_list_classifier_dic[asset_groups[:remove1]] == 'VRES':
                    for individual_assets in investment_list_dic_FP[asset_groups]:
                        if FP_combined_all_asset_capacity_dic[individual_assets] == 1:
                            CF = (df_tot_data_UC[individual_assets].sum() * (365 / clusters)) - FP_asset_OM_dic[
                                asset_groups[:remove1]]
                            df_asset_cf_UC.loc[year, individual_assets] = CF

                elif investment_list_classifier_dic[asset_groups[:remove1]] == 'Storage':
                    for individual_assets in investment_list_dic_FP[asset_groups]:
                        if FP_combined_all_asset_capacity_dic[individual_assets] == 1:
                            CF_charge = (df_tot_data_UC[individual_assets + '_ChargeRev'].sum()) * (365 / clusters)
                            CF_discharge = (df_tot_data_UC[individual_assets + '_DischargeRev'].sum()) * (
                                    365 / clusters)
                            CF = (CF_discharge - CF_charge) - FP_asset_OM_dic[asset_groups[:remove1]]
                            df_asset_cf_UC.loc[year, individual_assets] = CF

            # Do CM bids

            # D = df_rdsort_2['E demand'].max() + df_rdsort_2['H demand'].max()
            Epeak_y4 = df_rdsort_2['E demand'].max()
            Hpeak_y4 = df_rdsort_2['H demand'].max()
            Dpeak = Epeak_y4 + Hpeak_y4
            Dr = (Dpeak) * (1 + r)

            if capacity_mechanism == 1:

                # Chose NPV higher in loop
                df_asset_cm = pd.DataFrame()
                df_asset_cm_sorted = pd.DataFrame()

                counter = 0
                cm_assets = []

                for asset_groups in investment_list_dic_FP:
                    remove1 = len(asset_groups) - len('_asset_list')
                    for individual_assets in investment_list_dic_FP[asset_groups]:
                        if FP_combined_all_asset_capacity_dic[individual_assets] == 1:
                            if df_asset_cf_UC.loc[year, individual_assets] < 0:
                                counter += 1
                                cm_assets += [individual_assets]
                                df_asset_cm.loc[individual_assets, 'Volume bid'] = standard_capacity_dic[
                                    asset_groups[:remove1]]
                                df_asset_cm.loc[individual_assets, 'Price bid'] = (df_asset_cf_UC.loc[
                                                                                       year, individual_assets] * -1) / (
                                                                                      standard_capacity_dic[
                                                                                          asset_groups[:remove1]])
                                df_asset_cm_sorted = df_asset_cm.sort_values('Price bid')

                tot_volume = 0

                for asset in range(len(df_asset_cm_sorted)):
                    tot_volume += df_asset_cm_sorted.iloc[asset][0]
                    df_asset_cm_sorted.iloc[asset][0] = tot_volume

                if len(df_asset_cm_sorted) < 2:
                    if len(df_asset_cm_sorted) == 0:
                        df_asset_cm_sorted['Volume bid'] = 0
                        df_asset_cm_sorted['Price bid'] = 0
                    else:
                        df_asset_cm_sorted = df_asset_cm_sorted.append(df_asset_cm_sorted.iloc[0])
                        df_asset_cm_sorted.iloc[0, 0] = 0
                        df_asset_cm_sorted.iloc[0, 1] = df_asset_cm_sorted.iloc[1, 1]

                # df_asset_cm_sorted.plot(x='Volume bid', y='Price bid')
                # plt.ylabel('€/MWh')
                # plt.xlabel('MW')
                # plt.title('Capacity market bids in year' + str(year))
                # plt.show()

                if len(df_asset_cm_sorted) > 0:
                    if len(df_asset_cm_sorted) > 2:

                        # Input variables
                        demand_list = []
                        t_list = []
                        counter = 0
                        LM = 0
                        UM = 0
                        m = 0
                        x = 0
                        y = 0

                        # Calculations
                        UM = (Dpeak * (1 + r + um_in))
                        LM = (Dpeak * (1 + r - lm_in))
                        # m = -1 * (Pc / (LM - UM))  # slope
                        # b = (m * UM)

                        a = 0
                        b = 0

                        a = (0 - Pc) / (UM - LM)
                        b = Pc - (a * LM)

                        volume1 = df_asset_cm_sorted['Volume bid']
                        volume2 = np.linspace(0, LM, 500)
                        volume3 = np.linspace(LM, UM, 500)

                        m1, b1 = 0, Pc
                        m2, b2 = a, b

                        price1 = df_asset_cm_sorted['Price bid']
                        price2 = (volume2 * m1) + b1
                        price3 = (volume3 * m2) + b2

                        line_1 = LineString(np.column_stack((volume1, price1)))
                        line_2 = LineString(np.column_stack((volume2, price2)))
                        line_3 = LineString(np.column_stack((volume3, price3)))

                        volume_intersection = 0
                        price_intersection = 0

                        intersection = line_1.intersection(line_3)

                        if intersection.is_empty:
                            intersection = Point(0, 0)

                            intersection = line_1.intersection(line_2)
                            if intersection.is_empty:
                                intersection = Point(0, 0)

                                if df_asset_cm_sorted['Price bid'].min() < Pc:
                                    for index, row in df_asset_cm_sorted.iterrows():
                                        if df_asset_cm_sorted.loc[index, 'Price bid'] <= Pc:
                                            volume_intersection = volume_intersection + df_asset_cm_sorted.loc[
                                                index, 'Volume bid']
                                            price_intersection = df_asset_cm_sorted.loc[index, 'Price bid']
                                            if volume_intersection <= Dr:
                                                volume_intersection_actual = volume_intersection
                                                intersection = Point(volume_intersection_actual, price_intersection)

                        x, y = intersection.xy

                        # plt.plot(volume1, price1, 'blue')
                        # plt.plot(volume2, price2, 'black')
                        # plt.plot(volume3, price3, 'black')
                        # plt.plot(x, y, 'ro')
                        # plt.ylim(0, Pc * 1.1)
                        # plt.title('Capacity market in year' + str(year))
                        # plt.xlabel('MW')
                        # plt.ylabel('€/MWh')
                        #
                        # plt.show()

                        max_cm = x[0]

                        df_cm_cf = pd.DataFrame()

                        for index, row in df_asset_cm_sorted.iterrows():
                            if max_cm - df_asset_cm['Volume bid'][index] > 0:
                                max_cm = max_cm - df_asset_cm['Volume bid'][index]
                                CF = df_asset_cm['Volume bid'][index] * y[0]
                                df_cm_cf.loc[year, index] = CF
                            elif max_cm > 0:
                                CF = max_cm * y[0]
                                df_cm_cf.loc[year, index] = CF
                                max_cm = 0

                        for name, values in df_cm_cf.iteritems():
                            df_asset_cf_UC.loc[year, name] = df_asset_cf_UC.loc[year, name] + df_cm_cf.loc[
                                year, name]


                    else:
                        # Input variables
                        demand_list = []
                        t_list = []
                        counter = 0
                        LM = 0
                        UM = 0
                        m = 0
                        x = 0
                        y = 0

                        # Calculations
                        UM = (Dpeak * (1 + um_in))
                        LM = (Dpeak * (1 - lm_in))
                        # m = -1 * (Pc / (LM - UM))  # slope
                        # b = (m * UM)

                        a = 0
                        b = 0

                        a = (0 - Pc) / (UM - LM)
                        b = Pc - (a * LM)

                        volume1 = df_asset_cm_sorted['Volume bid']
                        volume2 = np.linspace(0, LM, 500)
                        volume3 = np.linspace(LM, UM, 500)

                        m1, b1 = 0, Pc
                        m2, b2 = a, b

                        price1 = df_asset_cm_sorted['Price bid']
                        price2 = (volume2 * m1) + b1
                        price3 = (volume3 * m2) + b2

                        line_1 = LineString(np.column_stack((volume1, price1)))
                        line_2 = LineString(np.column_stack((volume2, price2)))
                        line_3 = LineString(np.column_stack((volume3, price3)))

                        volume_intersection = 0
                        price_intersection = 0

                        intersection = line_1.intersection(line_3)

                        if intersection.is_empty:
                            intersection = Point(0, 0)

                            intersection = line_1.intersection(line_2)
                            if intersection.is_empty:
                                intersection = Point(0, 0)

                                if df_asset_cm_sorted['Price bid'].min() < Pc:
                                    for row in range(len(df_asset_cm_sorted.index)):
                                        if df_asset_cm_sorted.iloc[row, 1] <= Pc:
                                            volume_intersection = df_asset_cm_sorted.iloc[row, 0]
                                            if volume_intersection <= Dr:
                                                volume_intersection_actual = volume_intersection
                                                price_intersection = df_asset_cm_sorted.iloc[row, 1]
                                                intersection = Point(volume_intersection_actual, price_intersection)

                        x, y = intersection.xy

                        # plt.plot(volume1, price1, 'blue')
                        # plt.plot(volume2, price2, 'black')
                        # plt.plot(volume3, price3, 'red')
                        # plt.plot(x, y, 'ro')
                        # plt.ylim(0, Pc * 1.1)
                        # plt.title('Capacity market in year' + str(year))
                        # plt.xlabel('MW')
                        # plt.ylabel('€/MWh')
                        #
                        # plt.show()

                        max_cm = x[0]

                        df_cm_cf = pd.DataFrame()

                        for index, row in df_asset_cm_sorted.iterrows():
                            if max_cm - df_asset_cm['Volume bid'][index] > 0:
                                max_cm = max_cm - df_asset_cm['Volume bid'][index]
                                CF = df_asset_cm['Volume bid'][index] * y[0]
                                df_cm_cf.loc[year, index] = CF
                            elif max_cm > 0:
                                CF = max_cm * y[0]
                                df_cm_cf.loc[year, index] = CF
                                max_cm = 0

                        for name, values in df_cm_cf.iteritems():
                            df_asset_cf_UC.loc[year, name] = df_asset_cf_UC.loc[year, name] + df_cm_cf.loc[
                                year, name]

            # Chose NPV higher in loop
            df_asset_cf_FP = pd.DataFrame()
            df_asset_cf_FP1 = pd.DataFrame()
            df_asset_cf_FP = df_asset_cf_FP.append(df_asset_cf_UC)

            for x in range((asset_eco_lifetime_dic[change_asset_group])):
                df_asset_cf_FP1 = df_asset_cf_FP1.append(df_asset_cf_FP)
                df_asset_cf_FP1.reset_index(drop=True, inplace=True)

            df_asset_cf_FP1.loc[0, change_asset] = df_asset_cf_FP1.loc[0, change_asset] - (FP_investment_costs_dic[
                                                                                               change_asset_group] * equity_factor)

            df_asset_cf_FP1['debt'] = 1

            df_asset_cf_FP1[change_asset] = np.where(df_asset_cf_FP1['debt'] > 0,
                                                     df_asset_cf_FP1[change_asset] -
                                                     (((1 - equity_factor) * FP_investment_costs_dic[
                                                         change_asset_group]) / asset_eco_lifetime_dic[
                                                          change_asset_group]),
                                                     df_asset_cf_FP1[change_asset])

            npv = (npf.npv(0.09, df_asset_cf_FP1[change_asset])) / (standard_capacity_dic[change_asset_group])
            npv_dic_add = {change_asset: npv}
            npv_dic.update(npv_dic_add)

        max_npv_asset = getKeysByValue(npv_dic, max(npv_dic.values()))[0]

        if max(npv_dic.values()) <= 0:
            negative_npv = 1
        else:
            budget_number = budget_number - (FP_investment_costs_dic[RemoveNumber(max_npv_asset)] * equity_factor)
            delay_value = year + asset_delay_dic[RemoveNumber(max_npv_asset)]
            delay_dic[max_npv_asset] = delay_value

            for asset_groups in investment_list_dic_FP:
                remove1 = len(asset_groups) - len('_asset_list')
                if asset_groups[:remove1] == RemoveNumber(max_npv_asset):
                    new_asset = RemoveNumber(max_npv_asset) + str(len(investment_list_dic_FP[asset_groups]) + 1)
                    pipeline_list_dic[str(RemoveNumber(max_npv_asset)) + '_in_pipeline'] += 1
                    all_asset_age_dic[max_npv_asset] = asset_delay_dic[RemoveNumber(max_npv_asset)]
                    assets_this_round = [max_npv_asset] + assets_this_round

            for debt_years in range(year + 1 + asset_delay_dic[RemoveNumber(individual_assets)],
                                    (year) + asset_eco_lifetime_dic[RemoveNumber(individual_assets)]):
                df_debt.loc[debt_years, individual_assets] = FP_investment_costs_dic[
                                                                 RemoveNumber(individual_assets)] * (
                                                                         1 - equity_factor) / \
                                                             asset_eco_lifetime_dic[RemoveNumber(individual_assets)]

    ##############################################################################################
    #                                                                                            #
    #                       #CODE TO INITIATE DISMANTLE PART OF THE MIDO MODEL                   #
    #                                                                                            #
    ##############################################################################################

    dismantle_order_dic_sorted = {}
    dismantle_age_order_dic_sorted = {}

    for asset_groups in dismantle_list_dic:
        remove1 = len(asset_groups) - len('_asset_list')
        for individual_assets in dismantle_list_dic[asset_groups]:
            if all_asset_capacity_dic[individual_assets] == 1:
                if asset_lifetime_dic[asset_groups[:remove1]] == all_asset_age_dic[individual_assets]:
                    all_asset_capacity_dic[individual_assets] = 0
                    asset_capacity_dic[asset_groups[:remove1]] = asset_capacity_dic[asset_groups[:remove1]] - 1
                    investment_list_dic[asset_groups].remove(individual_assets)

    for asset_groups in dismantle_list_dic:
        remove1 = len(asset_groups) - len('_asset_list')
        for individual_assets in dismantle_list_dic[asset_groups]:
            if all_asset_capacity_dic[individual_assets] == 1:
                if year > lookbackperiod:
                    cf_in_lookback = df_dismantle.iloc[(year - lookbackperiod):(year)].sum()

                    if all_asset_age_dic[individual_assets] > lookbackperiod:
                        if cf_in_lookback[individual_assets] < 0:
                            dismantle_order_dic_sorted[individual_assets] = cf_in_lookback[individual_assets]

    dismantle_order_dic_sorted = dict(sorted(dismantle_order_dic_sorted.items(), key=lambda item: item[1]))

    for individual_assets in dismantle_order_dic_sorted:

        # Import weather + current e-demand + h-demand values
        df_onshore = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Onshore windpark output _std_weather_full_year.txt",
            names=['Onshore'])
        df_offshore = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Offshore windpark output_std_weather_full_year.txt",
            names=['Offshore'])
        df_solar = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Input weather during standard year\Solar output one asset_std_weather_full_year.txt",
            names=['Solar'])
        df_edemand = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Electricity demand_full_year.txt",
            names=['E demand'])
        df_hdemand = pd.read_csv(
            r"C:\Users\IEUser\Desktop\Yasin thesi files\Input data\Timeseries 8760\Hydrogen demand_full_year.txt",
            names=['H demand'])

        df_dismantle_min = pd.concat([df_onshore, df_offshore, df_solar, df_edemand, df_hdemand], axis=1)

        df_dismantle_min['Timeseries'] = 1

        df_dismantle_min['Onshore'] = np.where(df_dismantle_min['Timeseries'] > 0,
                                               df_dismantle_min['Onshore'] * (asset_capacity_dic['OnshoreWind']), 0)

        df_dismantle_min['Offshore'] = np.where(df_dismantle_min['Timeseries'] > 0,
                                                df_dismantle_min['Offshore'] * (asset_capacity_dic['OffshoreWind']), 0)

        df_dismantle_min['Solar'] = np.where(df_dismantle_min['Timeseries'] > 0,
                                             df_dismantle_min['Solar'] * (asset_capacity_dic['SolarPV']),
                                             0)

        df_dismantle_min['E demand'] = np.where(df_dismantle_min['Timeseries'] > 0,
                                                df_dismantle_min['E demand'] * (electricity_demand_change),
                                                0)

        df_dismantle_min['H demand'] = np.where(df_dismantle_min['Timeseries'] > 0,
                                                df_dismantle_min['H demand'] * (hydrogen_demand_change), 0)

        df_dismantle_min["Residual load"] = np.where(df_dismantle_min['Timeseries'] > 0,
                                                     (df_dismantle_min['H demand'] + df_dismantle_min['E demand']) - (
                                                                 df_dismantle_min['Onshore'] + df_dismantle_min[
                                                             'Offshore'] + df_dismantle_min['Solar']), 0)
        dismantle_D_residual = df_dismantle_min["Residual load"].max()

        total_capacity = 0
        dismantle_supply_ratio = 0

        for asset_groups_dismantle in investment_list_dic:
            remove2 = len(asset_groups_dismantle) - len('_asset_list')
            if investment_list_classifier_dic[asset_groups_dismantle[:remove2]] == 'Asset':
                standard_asset = standard_capacity_dic[asset_groups_dismantle[:remove2]]
                asset_capacity = asset_capacity_dic[asset_groups_dismantle[:remove2]]
                total_capacity += (standard_asset * asset_capacity)
            if investment_list_classifier_dic[asset_groups_dismantle[:remove2]] == 'Storage':
                standard_asset = standard_capacity_dic[asset_groups_dismantle[:remove2]]
                asset_capacity = asset_capacity_dic[asset_groups_dismantle[:remove2]]

                if asset_groups_dismantle == 'LiOnStorage_asset_list':
                    total_capacity += (standard_asset * 0.42) * (asset_capacity)
                if asset_groups_dismantle == 'ElectrolyserS_asset_list':
                    total_capacity += (standard_asset * 0.7) * (asset_capacity)
                if asset_groups_dismantle == 'ElectrolyserCCGT_asset_list':
                    total_capacity += (standard_asset * 0.57 * asset_capacity)
                if asset_groups_dismantle == 'ElectrolyserOCGT_asset_list':
                    total_capacity += (standard_asset * 0.39 * asset_capacity)

        dismantle_supply_ratio = (total_capacity) / dismantle_D_residual

        if dismantle_supply_ratio > dismantle_loop_stop:
            if individual_asset in government_asset_list:
                governmentcounter = 0
            else:
                all_asset_capacity_dic[individual_assets] = 0
                asset_capacity_dic[RemoveNumber(individual_assets)] = asset_capacity_dic[
                                                                          RemoveNumber(individual_assets)] - 1
                investment_list_dic[str(RemoveNumber(individual_assets)) + '_asset_list'].remove(individual_assets)

        # for asset_groups in dismantle_list_dic:
        # remove1 = len(asset_groups) - len('_asset_list')
        # for individual_assets in dismantle_list_dic[asset_groups]:
        # if all_asset_capacity_dic[individual_assets] == 1:
        #   if year > lookbackperiod:
        #      cf_in_lookback = df_dismantle.iloc[(year - lookbackperiod):(year)].sum()

        #    if all_asset_age_dic[individual_assets] > lookbackperiod:
        #         if cf_in_lookback[individual_assets] < 0:
        #          all_asset_capacity_dic[individual_assets] = 0
        #           asset_capacity_dic[asset_groups[:remove1]] = asset_capacity_dic[
        #                                                          asset_groups[:remove1]] - 1
        #        investment_list_dic[asset_groups].remove(individual_assets)

    install_asset_list = []
    install_asset_list = getKeysByValue(delay_dic, int(year))

    for individual_assets in install_asset_list:
        asset_capacity_dic[RemoveNumber(individual_assets)] += 1
        all_asset_capacity_dic[individual_assets] = 1
        all_asset_age_dic[individual_assets] = 0
        pipeline_list_dic[str(RemoveNumber(individual_assets)) + '_in_pipeline'] = pipeline_list_dic[str(RemoveNumber(
            individual_assets)) + '_in_pipeline'] - 1
        investment_list_dic[str(RemoveNumber(individual_assets)) + '_asset_list'] = [individual_assets] + \
                                                                                    investment_list_dic[
                                                                                        str(RemoveNumber(
                                                                                            individual_assets)) + '_asset_list']

    for individual_asset in list(delay_dic):
        if individual_asset in install_asset_list:
            del delay_dic[individual_asset]

    subsidiy_cost = 0

    #Install any government back assets if subsidies are part of the scenario

    government_counter_list = []

    government_all_dic = {**all_asset_capacity_dic, **delay_dic}

    for assets in investment_list:
        government_counter_list += [str(assets) + '_installed_counter']

    government_installed_counter_dic = {key: 0 for key in government_counter_list}

    for individual_assets in government_all_dic:
        for asset_groups in investment_list:
            if RemoveNumber(individual_assets) == asset_groups:
                government_installed_counter_dic[str(asset_groups) + '_installed_counter'] = \
                government_installed_counter_dic[str(asset_groups) + '_installed_counter'] + 1

    while government_installed_counter_dic['SolarPV_installed_counter'] < df_government_solar.iloc[year, 0]:
        government_asset = 'SolarPV' + str(government_installed_counter_dic['SolarPV_installed_counter'] + 1)
        asset_capacity_dic['SolarPV'] += 1
        all_asset_capacity_dic[government_asset] = 1
        all_asset_age_dic[government_asset] = 0
        government_installed_counter_dic['SolarPV_installed_counter'] += 1
        government_asset_list += [government_asset]
        subsidiy_cost += investment_costs_dic['SolarPV'] * 1

    while government_installed_counter_dic['OnshoreWind_installed_counter'] < df_goverment_onshore.iloc[year, 0]:
        government_asset = 'OnshoreWind' + str(government_installed_counter_dic['OnshoreWind_installed_counter'] + 1)
        asset_capacity_dic['OnshoreWind'] += 1
        all_asset_capacity_dic[government_asset] = 1
        all_asset_age_dic[government_asset] = 0
        government_installed_counter_dic['OnshoreWind_installed_counter'] += 1
        government_asset_list += [government_asset]
        subsidiy_cost += investment_costs_dic['OnshoreWind'] * 1

    while government_installed_counter_dic['OffshoreWind_installed_counter'] < df_government_offshore.iloc[year, 0]:
        government_asset = 'OffshoreWind' + str(government_installed_counter_dic['OffshoreWind_installed_counter'] + 1)
        asset_capacity_dic['OffshoreWind'] += 1
        all_asset_capacity_dic[government_asset] = 1
        all_asset_age_dic[government_asset] = 0
        government_installed_counter_dic['OffshoreWind_installed_counter'] += 1
        government_asset_list += [government_asset]
        subsidiy_cost += investment_costs_dic['OffshoreWind'] * 1


    year += 1