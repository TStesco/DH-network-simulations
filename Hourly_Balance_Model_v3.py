import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re
from datetime import datetime
from dateutil import parser

def runScenarios():
	cwd = os.getcwd()
	path = cwd + "/Hourly_Individual_Buildings/*.csv"

	# Start of scenario configuration
	# =====================================================================================================================================
	# define buildings in each scenario
	buildings_BC = ['S1', 'HC4', 'H2', 'H3', 'HC1', 'HC2', 'O2', 'O3', 'P2']
	buildings_ADJ =  ['S1', 'HC4', 'H2', 'H3', 'HC1', 'HC2', 'O2', 'O3', 'P2', 'HC5', 'HC3', 'H4', 'H1', 'O1', 'P1']
	buildings_EXT1 = ['S1', 'HC4', 'H2', 'H3', 'HC1', 'HC2', 'O2', 'O3', 'P2', 'HC5', 'HC3', 'H4', 'H1', 'O1', 'P1',
	                  'O4', 'O5', 'O6', 'O7', 'I1', 'I2', 'I3', 'O10']
	buildings_EXT2 = ['S1', 'HC4', 'H2', 'H3', 'HC1', 'HC2', 'O2', 'O3', 'P2', 'HC5', 'HC3', 'H4', 'H1', 'O1', 'P1',
	                  'O4', 'O5', 'O6', 'O7', 'I1', 'I2', 'I3','H6', 'H7', 'H8', 'H9', 'O11', 'O10']
	buildings_GSHP = ['S1', 'HC4', 'H2', 'H3', 'HC1', 'HC2', 'O2', 'O3', 'P2', 'HC5', 'HC3', 'H4', 'H1', 'O1', 'P1',
	                  'H6', 'H7', 'H8', 'H9,' 'O11', 'O4', 'O5', 'O6', 'O7', 'I1', 'I3', 'H5', 'HC8', 'HC6', 'HC7',                   
	                  'HC8', 'O8', 'O10', 'O9', 'I2']
	# create blank output dataframe
	all_columns = ['Zone Heating','Total Cooling','DHW', 'GSHP_MT', 'GSHP_LT', 'GSHP_HT', 'DHW_preheat', 'Biomass_HT_gen', 'Biomass_HT_use',
	               'process_HT_gen', 'process_MT_gen', 'process_HT_use','process_MT_use', 'SolarT_MT_gen',
	               'SolarT_MT_use', 'SolarT_HT_gen', 'SolarT_HT_use','MT_storage', 'MT_storage_use', 'HT_to_MT',
	               'Room Electricity', 'Lighting','MT_export', 'HT_export', 'LT_export', 'MT_temperature', 'HT_storage_use', 'HT_storage',
	               'HT_temperature', 'LT_storage_use', 'LT_storage', 'LT_temperature', 'HT_loss','MT_loss',
	               'LT_loss', 'HT_pump_elec', 'MT_pump_elec', 'LT_pump_elec']
	zero_demand = pd.DataFrame(np.zeros((8760, len(all_columns))), index=np.arange(1,8761), columns=all_columns)

	uncovered_columns = ['Buildingname','Total Cooling','DHW', 'ZH_oil', 'ZH_gas', 'ZH_GSHP', 'Room Electricity', 'Lighting']

	uncovered_demand = pd.DataFrame(columns=uncovered_columns)

	# constants across scenarios
	Elec_CO2_kWh = 0.1368 # 0.038(kg/MJ)/0.277778(MJ/kWh) # kg-CO2-eq/kWh CH-verbrauchermix from 170920_AS_Areal_Boundary_Conditions.xlsx
	Elec_CO2_kWh_2050 = 0.0900 # 0.025(kg/MJ)/0.277778(MJ/kWh) from NEP scenario
	Gas_CO2_kWh = 0.3132 # 0.087/0.277778 # kg-CO2-eq/kWh Heizzentrale gas from 170920_AS_Areal_Boundary_Conditions.xlsx
	Oil_CO2_kWh = 0.4032 # 0.112/0.277778 # kg-CO2-eq/kWh Heizzentrale oel from 170920_AS_Areal_Boundary_Conditions.xlsx
	hydro_production = 2706166
	oil_n_th = 0.7
	gas_n_th = 0.8
	dec_GSHP_COP_H = 3.45
	dec_GSHP_COP_C = 3.45
	SolarT_HT_coeff = 0.5

	# scenario parameters
	scenarios = {0:{'data':zero_demand.copy(),'buildings':buildings_BC, 'uncovered_demand': uncovered_demand,
	                'params':{'GSHP_MT_cap':3000,
	                          'GSHP_LT_cap':-2000,
	                          'GSHP_COP_H':3.74,
	                          'GSHP_COP_C':9.6,
	                          'oil_n_th': oil_n_th,
	                          'gas_n_th': gas_n_th,
	                          'dec_GSHP_COP_H': dec_GSHP_COP_H,
	                          'dec_GSHP_COP_C': dec_GSHP_COP_C,
	                          'HT_storage_vol':165.20, # m3
	                          'MT_storage_vol':241.38,
	                          'LT_storage_vol':260.29,
	                          'HT_storage_max_T':95,
	                          'HT_storage_min_T':80,
	                          'MT_storage_max_T':95,
	                          'MT_storage_min_T':60,
	                          'LT_storage_max_T':16,
	                          'LT_storage_min_T':4,
	                          'Elec_CO2_kWh': Elec_CO2_kWh,
	                          'Elec_CO2_kWh_2050': Elec_CO2_kWh_2050,
	                          'Gas_CO2_kWh': Gas_CO2_kWh,
	                          'Oil_CO2_kWh': Oil_CO2_kWh,
	                          'hydro_production':hydro_production}
	               },
	             1:{'data':zero_demand.copy(),'buildings':buildings_ADJ, 'uncovered_demand': uncovered_demand,
	                 'params':{'GSHP_MT_cap':5000,
	                          'GSHP_LT_cap':-2000,
	                          'GSHP_COP_H':3.74,
	                          'GSHP_COP_C':9.6,
	                          'oil_n_th': oil_n_th,
	                          'gas_n_th': gas_n_th,
	                          'dec_GSHP_COP_H': dec_GSHP_COP_H,
	                          'dec_GSHP_COP_C': dec_GSHP_COP_C,
	                          'HT_storage_vol':165.56,
	                          'MT_storage_vol':243.30,
	                          'LT_storage_vol':261.71,
	                          'HT_storage_max_T':95,
	                          'HT_storage_min_T':80,
	                          'MT_storage_max_T':95,
	                          'MT_storage_min_T':60,
	                          'LT_storage_max_T':16,
	                          'LT_storage_min_T':4,
	                          'Elec_CO2_kWh': Elec_CO2_kWh,
	                          'Elec_CO2_kWh_2050': Elec_CO2_kWh_2050,
	                          'Gas_CO2_kWh': Gas_CO2_kWh,
	                          'Oil_CO2_kWh': Oil_CO2_kWh,
	                          'hydro_production':hydro_production}
	               },
	             2:{'data':zero_demand.copy(),'buildings':buildings_EXT1, 'uncovered_demand': uncovered_demand,
	                 'params':{'GSHP_MT_cap':6000,
	                          'GSHP_LT_cap':-2500,
	                          'GSHP_COP_H':3.74,
	                          'GSHP_COP_C':9.6,
	                          'oil_n_th': oil_n_th,
	                          'gas_n_th': gas_n_th,
	                          'dec_GSHP_COP_H': dec_GSHP_COP_H,
	                          'dec_GSHP_COP_C': dec_GSHP_COP_C,
	                          'HT_storage_vol':240.21,
	                          'MT_storage_vol':353.24,
	                          'LT_storage_vol':378.42,
	                          'HT_storage_max_T':95,
	                          'HT_storage_min_T':80,
	                          'MT_storage_max_T':95,
	                          'MT_storage_min_T':60,
	                          'LT_storage_max_T':16,
	                          'LT_storage_min_T':4,
	                          'Elec_CO2_kWh': Elec_CO2_kWh,
	                          'Elec_CO2_kWh_2050': Elec_CO2_kWh_2050,
	                          'Gas_CO2_kWh': Gas_CO2_kWh,
	                          'Oil_CO2_kWh': Oil_CO2_kWh,
	                          'hydro_production':hydro_production}
	               },
	             3:{'data':zero_demand.copy(),'buildings':buildings_EXT2, 'uncovered_demand': uncovered_demand,
	                 'params':{'GSHP_MT_cap':7000,
	                          'GSHP_LT_cap':-2500,
	                          'GSHP_COP_H':3.74,
	                          'GSHP_COP_C':9.6,
	                          'oil_n_th': oil_n_th,
	                          'gas_n_th': gas_n_th,
	                          'dec_GSHP_COP_H': dec_GSHP_COP_H,
	                          'dec_GSHP_COP_C': dec_GSHP_COP_C,
	                          'HT_storage_vol':350.97,
	                          'MT_storage_vol':514.61,
	                          'LT_storage_vol':550.79,
	                          'HT_storage_max_T':95,
	                          'HT_storage_min_T':80,
	                          'MT_storage_max_T':95,
	                          'MT_storage_min_T':60,
	                          'LT_storage_max_T':16,
	                          'LT_storage_min_T':4,
	                          'Elec_CO2_kWh': Elec_CO2_kWh,
	                          'Elec_CO2_kWh_2050': Elec_CO2_kWh_2050,
	                          'Gas_CO2_kWh': Gas_CO2_kWh,
	                          'Oil_CO2_kWh': Oil_CO2_kWh,
	                          'hydro_production':hydro_production}
	               },
	            }
	# End of scenario configuration
	# =====================================================================================================================================

	# read pvt production to memory
	csv_pvt = pd.read_csv(cwd+'/pvt_production.csv')
	pvt_df = csv_pvt.reindex(index=np.arange(1,8761))
	pvt_df = pvt_df.apply(lambda x: x*SolarT_HT_coeff)

	# loop over .csv files and sum them according to scenarios
	for fname in glob.glob(path):
	    # read csv
	    csv_df = pd.read_csv(fname, encoding = "ISO-8859-1")
	    hourly_df = csv_df[1:csv_df.shape[0]] #remove units row
	    #hourly_df = hourly_df.drop(labels='Unnamed: 23', axis=1) #remove padding
	    building_name = re.search('/Hourly_Individual_Buildings/(.*)_hourly.csv', fname).group(1)
	    
	    # add solar thermal generation to hourly_df
	    #add_pvt_to_hourly(pvt_df, hourly_df, building_name)
	    #hourly_df[:,'SolarT_HT_gen'] = pvt_df[building_name]
	    if building_name in list(pvt_df):
	        hourly_df = hourly_df.assign(SolarT_HT_gen=pvt_df[building_name])
	    else: 
	        hourly_df = hourly_df.assign(SolarT_HT_gen=np.zeros((hourly_df.shape[0], 1)))
	        
	    # check if building is part of scenario
	    for n in range(len(scenarios)):
	        if building_name in scenarios[n]['buildings']:
	            add_to_scenario(scenarios[n]['data'], hourly_df, building_name)
	        else:
	            scenarios[n]['uncovered_demand'] = add_to_uncovered_demand(scenarios[n]['uncovered_demand'], hourly_df, building_name)


	hours = np.arange(1,8761)
	for n in range(len(scenarios)):
	    add_biomass(scenarios[n]['data'])
	    add_thermal_bath(scenarios[n]['data'])
	    balance_hours_3s(hours, scenarios[n]['data'], scenarios[n]['params'], n)

	# thermal export chart
	fig, ax = plt.subplots(figsize=(20,10))
	xaxis = range(1,25)
	for i in range(len(scenarios)):
	    hourly_totals = []
	    for n in range(1,25):
	        hour_day_list = np.arange(n, 8760, 24)
	        hourly_totals.append((scenarios[i]['data']['HT_export'].iloc[hour_day_list].sum()+                              scenarios[i]['data']['MT_export'].iloc[hour_day_list].sum())/1000)
	    ax.plot(xaxis, hourly_totals)

	plt.ylim(0, 2000)
	plt.xlim(1, 24)
	plt.xticks(np.arange(1,24,2))
	ax.set_ylabel('Thermal Exports (MWh/a)', fontsize=18)
	ax.set_xlabel('Hour of Day', fontsize=18)
	# Set the chart's title
	ax.set_title('Annual Thermal Exports by Hour of Day', fontsize=20)
	plt.legend(['Scenario 0','Scenario 1','Scenario 2','Scenario 3',], loc='upper left')
	plt.savefig('v3_plot_hours_thermal_export')

	# DHW demand chart
	fig, ax = plt.subplots(figsize=(20,10))
	xaxis = range(1,25)
	for i in range(len(scenarios)):
	    hourly_totals = []
	    for n in range(1,25):
	        hour_day_list = np.arange(n, 8760, 24)
	        hourly_totals.append(scenarios[i]['data']['DHW'].iloc[hour_day_list].sum()/1000)
	    ax.plot(xaxis, hourly_totals)

	plt.ylim(0, 1800)
	plt.xlim(1, 24)
	plt.xticks(np.arange(1,24,2))
	ax.set_ylabel('DHW Demand (MWh/a)', fontsize=18)
	ax.set_xlabel('Hour of Day', fontsize=18)
	# Set the chart's title
	ax.set_title('Annual DHW Demand by Hour of Day', fontsize=20)
	plt.legend(['Scenario 0','Scenario 1','Scenario 2','Scenario 3',], loc='upper left')
	plt.savefig('v3_plot_hours_DHW_Demand')
	
	# generate code for sankey diagrams
	for n in range(len(scenarios)):
		gen_sankey_input(scenarios[n], n)

	plot_scenario_characteristics(scenarios, 'v3_plot_cap_characteristics.png')
	plot_scenario_COP(scenarios, 'v3_plot_sys_COP.png')
	plot_scenarios_CO2(scenarios, 'v3_plot_scenarios_CO2.png')
	plot_district_CO2(scenarios, 'v3_plot_district_CO2.png')
	plot_process_use(scenarios, 'v3_plot_process_use.png')
	plot_exports(scenarios, 'v3_plot_exports.png')
	plot_2000W_watt_goals(scenarios, 'v3_plot_2000W_watt_goals')
	plot_2000W_co2_goals(scenarios, 'v3_plot_2000W_co2_goals')


def I_process_schedule(val,dt):
    day_series = [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0] # 17 hr per day
    day_series = np.array(day_series, dtype=int)*val
    weekend_series = np.zeros(24*2)
    week_series = day_series.copy()
    for n in range(4):
        week_series = np.concatenate((week_series, day_series))

    week_series = np.concatenate((week_series,weekend_series))

    hour_of_week = dt.isoweekday()*24+dt.hour
    year_series = week_series[hour_of_week:len(week_series)]
    for n in range(52):
        year_series = np.concatenate((year_series, week_series))

    year_series = year_series[0:8760]
    return year_series

def add_biomass(out_df):
    annual_production = 1016000
    hourly_production = (annual_production/8760)*(24/17) # annual / hours p year * (hours p day / hours p day)
    biomass_array = I_process_schedule(hourly_production, parser.parse('01.01.2002 01:00:00'))
    Biomass_HT_gen = pd.DataFrame(biomass_array, index=out_df.index, columns=['Biomass_HT_gen'])
    out_df['Biomass_HT_gen'] = out_df['Biomass_HT_gen'].add(pd.to_numeric(Biomass_HT_gen['Biomass_HT_gen'], errors='coerce'))

def add_to_scenario(out_df,add_df, building_name):
    
    # DHW conversion from DB output to kWh standard
    dhw_coeff = 0.85
    process_n_hx = 0.9 
    
    # add zeros total cooling column if not existing
    if not('Total Cooling' in add_df):
        cooling = pd.DataFrame(np.zeros((out_df.shape[0], 1)), index=out_df.index, columns=['Total Cooling'])
        add_df = pd.concat([add_df,cooling['Total Cooling']], axis=1)
        
    # add zeros 'DHW (Electricity)' column if not existing
    if not('DHW (Electricity)' in add_df):
        cooling = pd.DataFrame(np.zeros((out_df.shape[0], 1)), index=out_df.index, columns=['DHW (Electricity)'])
        add_df = pd.concat([add_df,cooling['DHW (Electricity)']], axis=1)
        
    process_heat = {'I1':405*process_n_hx,
                    'I2':104.496*process_n_hx,
                    'I3':1890*process_n_hx,
                    'O2':297.5*process_n_hx,
                    'O10':170*process_n_hx}   
    
    if building_name in process_heat:
        # handle MT process heat
        if building_name in ['I3', 'O2', 'O10']:
            if building_name in ['I3']:
                process_array = I_process_schedule(process_heat[building_name], parser.parse(add_df['Date/Time'][1]))
            else: # 24/7 server rooms
                process_array = np.full((out_df.shape[0], 1), process_heat[building_name])
            process_MT_gen = pd.DataFrame(process_array, index=out_df.index, columns=['process_MT_gen'])
            process_HT_gen = pd.DataFrame(np.zeros((out_df.shape[0], 1)), index=out_df.index, columns=['process_HT_gen'])
        # handle HT process heat
        elif building_name in ['I1','I2']:
            process_array = I_process_schedule(process_heat[building_name], parser.parse(add_df['Date/Time'][1]))
            process_HT_gen = pd.DataFrame(process_array, index=out_df.index, columns=['process_HT_gen'])
            process_MT_gen = pd.DataFrame(np.zeros((out_df.shape[0], 1)), index=out_df.index, columns=['process_MT_gen'])
    else:
        process_MT_gen = pd.DataFrame(np.zeros((out_df.shape[0], 1)), index=out_df.index, columns=['process_MT_gen'])
        process_HT_gen = pd.DataFrame(np.zeros((out_df.shape[0], 1)), index=out_df.index, columns=['process_HT_gen'])

    # elementwise add to out_df
    out_df['Zone Heating'] = out_df['Zone Heating'].add(pd.to_numeric(add_df['Zone Heating'], errors='coerce'))
    out_df['Room Electricity'] = out_df['Room Electricity'].add(pd.to_numeric(add_df['Room Electricity'], errors='coerce'))
    out_df['Lighting'] = out_df['Lighting'].add(pd.to_numeric(add_df['Lighting'], errors='coerce'))
    out_df['Total Cooling'] = out_df['Total Cooling'].add(pd.to_numeric(add_df['Total Cooling'], errors='coerce'))
    out_df['DHW'] = out_df['DHW'].add(dhw_coeff*pd.to_numeric(add_df['DHW (Electricity)'], errors='coerce'))
    out_df['process_HT_gen'] = out_df['process_HT_gen'].add(pd.to_numeric(process_HT_gen['process_HT_gen'], errors='coerce'))
    out_df['process_MT_gen'] = out_df['process_MT_gen'].add(pd.to_numeric(process_MT_gen['process_MT_gen'], errors='coerce'))
    out_df['SolarT_HT_gen'] = out_df['SolarT_HT_gen'].add(pd.to_numeric(add_df['SolarT_HT_gen'], errors='coerce'))

def add_to_uncovered_demand(out_df, add_df, building_name):
    
    # DHW conversion from DB output to kWh standard
    dhw_coeff = 0.85
    
    # settings for existing building heating fuel
    has_oil = ['H1', 'H4', 'HC3', 'P1', 'O5', 'HC5', 'O11', 'I3']
    has_gas = ['O1', 'O4', 'O6', 'O7', 'I1', 'H6', 'H7', 'H8', 'H9']
    has_GSHP = ['H5', 'HC6', 'HC7', 'HC8', 'O8', 'O10', 'O9', 'I2']
    ZH_oil = 0
    ZH_gas = 0
    ZH_GSHP = 0
    DHW = 0
    total_cooling = 0
    
    # add total cooling column if existing
    if 'Total Cooling' in add_df:
        total_cooling = pd.to_numeric(add_df['Total Cooling'], errors='coerce').sum()
        
    # add 'DHW (Electricity)' column if existing
    if 'DHW (Electricity)' in add_df:
        DHW = pd.to_numeric(add_df['DHW (Electricity)'], errors='coerce').sum()
        
    if building_name in has_oil:
        ZH_oil = pd.to_numeric(add_df['Zone Heating'], errors='coerce').sum()
    elif building_name in has_gas:
        ZH_gas = pd.to_numeric(add_df['Zone Heating'], errors='coerce').sum()
    elif building_name in has_GSHP:
        ZH_GSHP = pd.to_numeric(add_df['Zone Heating'], errors='coerce').sum()
        
    Room_Electricity = pd.to_numeric(add_df['Room Electricity'], errors='coerce').sum()
    Lighting = pd.to_numeric(add_df['Lighting'], errors='coerce').sum()

    uncovered_columns = ['Buildingname','Total Cooling','DHW', 'ZH_oil', 'ZH_gas', 'ZH_GSHP', 'Room Electricity', 'Lighting']
    # elementwise add to out_df
    append_df = pd.DataFrame([[building_name,                               total_cooling,                               DHW,                               ZH_oil,                               ZH_gas,                              ZH_GSHP,                              Room_Electricity,                              Lighting]]                              , columns=uncovered_columns)
    return out_df.append(append_df)

def add_thermal_bath(out_df):
    daily_heating = 12801 # kWh
    dt = parser.parse('01.01.2002 01:00:00')
    
    weekday_series = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0] # 7-16h heating
    weekday_series = np.array(weekday_series, dtype=int)*(daily_heating/sum(weekday_series))
    
    weekend_series = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0] # 7-16h heating
    weekend_series = np.array(weekend_series, dtype=int)*(daily_heating/sum(weekday_series))
    
    week_series = weekday_series.copy()
    for n in range(4):
        week_series = np.concatenate((week_series, weekday_series))

    for n in range(2):
        week_series = np.concatenate((week_series,weekend_series))

    hour_of_week = dt.isoweekday()*24+dt.hour
    year_series = week_series[hour_of_week:len(week_series)]
    for n in range(52):
        year_series = np.concatenate((year_series, week_series))

    thermal_bath_array = year_series[0:8760]
    thermal_bath = pd.DataFrame(thermal_bath_array, index=out_df.index, columns=['DHW'])
    out_df['DHW'] = out_df['DHW'].add(pd.to_numeric(thermal_bath['DHW'], errors='coerce'))

def get_th_losses (scenario_num):
    # format is MT_loss, HT_loss, LT_loss each in kW/K
    net_losses = {0:[0.731,0.600,0.744],
                  1:[0.863,0.686,0.839],
                  2:[1.240,0.944,1.137],
                  3:[1.655,1.293,1.529]}
    return net_losses[scenario_num]

def get_pump_elec (scenario_num):
    pump_elec = {0:[1.3, 0.47, 0.99],
                  1:[1.83, 0.77, 1.84],
                  2:[2.47, 0.98, 2.34],
                  3:[2.87, 1.15, 2.44]}
    return pump_elec[scenario_num]

def balance_hours_3s (hours, out_df, params, scenario_num):
    
    # get thermal loss rates
    MT_loss_rate, HT_loss_rate, LT_loss_rate = get_th_losses(scenario_num)
    
    # get pump elec
    HT_pump_elec, MT_pump_elec, LT_pump_elec = get_pump_elec(scenario_num)
    
    # initialize parameters
    GSHP_MT = 0
    GSHP_LT = 0
    GSHP_MT_cap_param = params['GSHP_MT_cap']
    GSHP_LT_cap_param = params['GSHP_LT_cap']
    HT_storage = 0
    HT_storage_vol = params['HT_storage_vol']
    HT_storage_max_T = params['HT_storage_max_T']
    HT_storage_min_T = params['HT_storage_min_T']
    HT_temperature = 0
    HT_storage_demand = 0
    SolarT_HT_gen = 0
    MT_storage = 0
    MT_storage_vol = params['MT_storage_vol']
    MT_storage_max_T = params['MT_storage_max_T']
    MT_storage_min_T = params['MT_storage_min_T']
    MT_temperature = 0
    MT_storage_demand = 0
    SolarT_MT_gen = 0
    MT_storage_use = 0
    HT_storage_use = 0
    LT_storage = 0
    LT_storage_vol = params['LT_storage_vol']
    LT_storage_max_T = params['LT_storage_max_T']
    LT_storage_min_T = params['LT_storage_min_T']
    LT_storage_demand = 0
    LT_temperature = 0
    LT_storage_use = 0
    ground_T = 10
    mains_water_T = 10
    DHW_preheat_T = 40
    DHW_T = 60
    # from https://www.engineeringtoolbox.com/water-thermal-properties-d_162.html
    HT_cap_water = 1.1335*HT_storage_vol # kWh/m3 = 4.198 (kJ/kg.K) * 972 (kg/m3) * 0.000277778 (kWh/kJ)
    MT_cap_water = 1.1427*MT_storage_vol # kWh/m3 = 4.185 (kJ/kg.K) * 983 (kg/m3) * 0.000277778 (kWh/kJ)
    LT_cap_water = 1.1681*LT_storage_vol # kWh/m3 = 4.205 (kJ/kg.K) * 1000 (kg/m3) * 0.000277778 (kWh/kJ)
    
    for hour in hours:
        
        # reset counter vars
        GSHP_HT = 0
        GSHP_MT = 0
        GSHP_LT = 0
        process_HT_use = 0
        process_MT_use = 0
        HT_storage_use = 0
        MT_storage_use = 0
        LT_storage_use = 0
        SolarT_HT_use = 0
        SolarT_MT_use = 0
        Biomass_HT_use = 0
        HT_to_MT = 0
        
        # HT network balance      
        
        if (hour == 1):
            HT_storage = 0
            HT_temperature = HT_storage_min_T
        else:
            HT_storage = out_df.loc[hour-1]['HT_storage']
            HT_temperature = out_df.loc[hour-1]['HT_temperature']
        
        HT_loss = HT_loss_rate*(HT_temperature-ground_T)
        process_HT_gen = out_df.loc[hour]['process_HT_gen']
        SolarT_HT_gen =  out_df.loc[hour]['SolarT_HT_gen']
        Biomass_HT_gen = out_df.loc[hour]['Biomass_HT_gen']
        
        # balance demand
        
        # pre-heating
        #from IPython.core.debugger import Tracer; Tracer()()
        n_preheat_hx = 0.9
        DHW_preheat = (DHW_preheat_T - mains_water_T)/(DHW_T - mains_water_T)*out_df.loc[hour]['DHW']
        demand_DHW = out_df.loc[hour]['DHW'] - DHW_preheat/n_preheat_hx + HT_loss
        
        if (demand_DHW < process_HT_gen):
            process_HT_use = demand_DHW
            demand_DHW = 0
        else:
            process_HT_use = process_HT_gen
            demand_DHW = demand_DHW - process_HT_use
            
        if (demand_DHW < SolarT_HT_gen):
            SolarT_HT_use = demand_DHW
            demand_DHW = 0
        else:
            SolarT_HT_use = SolarT_HT_gen
            demand_DHW = demand_DHW - SolarT_HT_use
            
        if (demand_DHW < Biomass_HT_gen):
            Biomass_HT_use = demand_DHW
            demand_DHW = 0
        else:
            Biomass_HT_use = Biomass_HT_gen
            demand_DHW = demand_DHW - Biomass_HT_use

        if (demand_DHW < HT_storage):
            HT_storage_use = demand_DHW
            HT_storage = HT_storage - HT_storage_use
            HT_temperature += - HT_cap_water*HT_storage_use
            demand_DHW = 0
        else:
            HT_storage_use = HT_storage
            demand_DHW = demand_DHW - HT_storage_use
            HT_storage = 0
            HT_temperature += - HT_cap_water*HT_storage_use
            
        if (demand_DHW > 0):
            GSHP_HT = demand_DHW
            demand_DHW = 0
        
        # HT storage balance
        
        HT_storage_demand = HT_cap_water*(HT_storage_max_T-HT_storage_min_T) - HT_storage
        if (HT_storage_demand>0):
            if (HT_storage_demand<(process_HT_gen-process_HT_use)):
                process_HT_use = process_HT_use + HT_storage_demand
                HT_storage_demand = 0
            else:
                HT_storage_demand = HT_storage_demand - (process_HT_gen-process_HT_use)
                process_HT_use = process_HT_gen

            if (HT_storage_demand<(SolarT_HT_gen-SolarT_HT_use)):
                SolarT_HT_use = SolarT_HT_use + HT_storage_demand
                HT_storage_demand = 0
            else:
                HT_storage_demand = HT_storage_demand - (SolarT_HT_gen-SolarT_HT_use)
                SolarT_HT_use = SolarT_HT_gen
            HT_temperature = (HT_cap_water*(HT_storage_max_T-HT_storage_min_T) - HT_storage_demand)/             (HT_cap_water*(HT_storage_max_T-HT_storage_min_T))*(HT_storage_max_T-HT_storage_min_T)+HT_storage_min_T
        
        out_df.loc[hour]['GSHP_HT'] = GSHP_HT
        out_df.loc[hour]['HT_storage_use'] = HT_storage_use
        out_df.loc[hour]['HT_temperature'] = HT_temperature
        out_df.loc[hour]['HT_storage'] = HT_cap_water*(HT_storage_max_T-HT_storage_min_T) - HT_storage_demand
        out_df.loc[hour]['Biomass_HT_use'] = Biomass_HT_use
        
        # zone heating balance
                     
        if (hour == 1):
            MT_storage = 0
            MT_temperature = MT_storage_min_T
        else:
            MT_storage = out_df.loc[hour-1]['MT_storage']
            MT_temperature = out_df.loc[hour-1]['MT_temperature']
        
        MT_loss = MT_loss_rate*(MT_temperature-ground_T)
        demand_ZH = out_df.loc[hour]['Zone Heating'] + MT_loss + DHW_preheat
        process_MT_gen = out_df.loc[hour]['process_MT_gen']  
        SolarT_MT_gen = out_df.loc[hour]['SolarT_MT_gen']

        #from IPython.core.debugger import Tracer; Tracer()()
        if (demand_ZH < process_MT_gen):
            process_MT_use = demand_ZH
            demand_ZH = 0
        else:
            process_MT_use = process_MT_gen
            demand_ZH = demand_ZH - process_MT_use
            
        if (demand_ZH < SolarT_MT_gen):
            SolarT_MT_use = demand_ZH
            demand_ZH = 0
        else:
            SolarT_MT_use = SolarT_MT_gen
            demand_ZH = demand_ZH - SolarT_MT_use
        
        if (demand_ZH>0):
            # add in excess HT network
            if (demand_ZH < (process_HT_gen-process_HT_use)):
                process_HT_use = process_HT_use + demand_ZH
                HT_to_MT += demand_ZH
                demand_ZH = 0
            else:
                demand_ZH = demand_ZH - (process_HT_gen-process_HT_use)
                HT_to_MT += process_HT_gen-process_HT_use
                process_HT_use = process_HT_gen

            if (demand_ZH < (SolarT_HT_gen-SolarT_HT_use)):
                SolarT_HT_use = SolarT_HT_use + demand_ZH
                HT_to_MT += demand_ZH
                demand_ZH = 0
            else:
                demand_ZH = demand_ZH - (SolarT_HT_gen-SolarT_HT_use)
                HT_to_MT += SolarT_HT_gen-SolarT_HT_use
                SolarT_HT_use = SolarT_HT_gen
            
            # MT storage use
            if (demand_ZH < MT_storage):
                MT_storage_use = demand_ZH
                MT_storage = MT_storage - MT_storage_use
                MT_temperature = MT_temperature - MT_cap_water*MT_storage_use
                demand_ZH = 0
            else:
                MT_storage_use = MT_storage
                demand_ZH = demand_ZH - MT_storage_use
                MT_storage = 0
                MT_temperature = MT_temperature - MT_cap_water*MT_storage_use

            if (demand_ZH > 0):
                GSHP_MT = demand_ZH
                demand_ZH = 0
            
        # MT storage balance
        
        MT_storage_demand = MT_cap_water*(MT_storage_max_T-MT_storage_min_T) - MT_storage
        if (MT_storage_demand>0):
            if (MT_storage_demand<(process_MT_gen-process_MT_use)):
                process_MT_use = process_MT_use + MT_storage_demand
                MT_storage_demand = 0
            else:
                MT_storage_demand = MT_storage_demand - (process_MT_gen-process_MT_use)
                process_MT_use = process_MT_gen

            if (MT_storage_demand<(SolarT_MT_gen-SolarT_MT_use)):
                SolarT_MT_use = SolarT_MT_use + MT_storage_demand
                MT_storage_demand = 0
            else:
                MT_storage_demand = MT_storage_demand - (SolarT_MT_gen-SolarT_MT_use)
                SolarT_MT_use = SolarT_MT_gen
            
            # add excess HT network
            if (MT_storage_demand<(process_HT_gen-process_HT_use)):
                process_HT_use = process_HT_use + MT_storage_demand
                HT_to_MT += MT_storage_demand
                MT_storage_demand = 0
            else:
                MT_storage_demand = MT_storage_demand - (process_HT_gen-process_HT_use)
                HT_to_MT += (process_HT_gen-process_HT_use)
                process_HT_use = process_HT_gen

            if (MT_storage_demand<(SolarT_HT_gen-SolarT_HT_use)):
                SolarT_HT_use = SolarT_HT_use + MT_storage_demand
                HT_to_MT += MT_storage_demand
                MT_storage_demand = 0
            else:
                MT_storage_demand = MT_storage_demand - (SolarT_HT_gen-SolarT_HT_use)
                HT_to_MT += (SolarT_HT_gen-SolarT_HT_use)
                SolarT_HT_use = SolarT_HT_gen
                
            MT_temperature = (MT_cap_water*(MT_storage_max_T-MT_storage_min_T) - MT_storage_demand)/             (MT_cap_water*(MT_storage_max_T-MT_storage_min_T))*(MT_storage_max_T-MT_storage_min_T)+MT_storage_min_T
            
        out_df.loc[hour]['DHW_preheat'] = DHW_preheat
        out_df.loc[hour]['HT_to_MT'] = HT_to_MT
        out_df.loc[hour]['MT_temperature'] = MT_temperature
        out_df.loc[hour]['MT_storage'] = MT_cap_water*(MT_storage_max_T-MT_storage_min_T) - MT_storage_demand
        out_df.loc[hour]['MT_storage_use'] = MT_storage_use
        out_df.loc[hour]['process_MT_use'] = process_MT_use   
        out_df.loc[hour]['GSHP_MT'] = GSHP_MT
        out_df.loc[hour]['process_HT_use'] = process_HT_use
        out_df.loc[hour]['SolarT_HT_use'] = SolarT_HT_use
        
        # Cooling balance

        if (hour == 1):
            LT_storage = 0
            LT_temperature = LT_storage_min_T
        else:
            LT_storage = out_df.loc[hour-1]['LT_storage']
            LT_temperature = out_df.loc[hour-1]['LT_temperature']
            if (LT_storage>0):
                print('Hour: %s LT_storage above zero %s' % (hour, LT_storage))
        
        LT_loss = LT_loss_rate*(LT_temperature-ground_T)
        demand_C = out_df.loc[hour]['Total Cooling'] + LT_loss
        
        if (demand_C<0):
            GSHP_LT = demand_C
            demand_C = 0
       
        out_df.loc[hour]['GSHP_LT'] = GSHP_LT
        out_df.loc[hour]['LT_storage'] = LT_storage #LT_storage_cap - LT_storage_demand
        out_df.loc[hour]['LT_storage_use'] = LT_storage_use
        out_df.loc[hour]['LT_temperature'] = LT_temperature
        out_df.loc[hour]['HT_loss'] = HT_loss
        out_df.loc[hour]['MT_loss'] = MT_loss
        out_df.loc[hour]['LT_loss'] = LT_loss
        
        # pump energy
        out_df.loc[hour]['HT_pump_elec'] = HT_pump_elec
        out_df.loc[hour]['MT_pump_elec'] = MT_pump_elec
        out_df.loc[hour]['LT_pump_elec'] = LT_pump_elec
        
        # Excess thermal export
        out_df.loc[hour]['HT_export'] = (SolarT_HT_gen-SolarT_HT_use) + (process_HT_gen-process_HT_use)
        out_df.loc[hour]['MT_export'] = (SolarT_MT_gen-SolarT_MT_use) + (process_MT_gen-process_MT_use)

def plot_scenario_characteristics(scenarios, filename):

    scenario_names = []
    process_MT_use = []
    process_HT_use = []
    SolarT_HT_use = []
    GSHP_HT_cap = []
    GSHP_MT_cap = []
    GSHP_LT_cap = []
    HT_storage_cap = []
    MT_storage_cap = []

    for n in range(len(scenarios)):
        scenario_names.append(n)
        process_MT_use.append(scenarios[n]['data']['process_MT_use'].max())
        process_HT_use.append(scenarios[n]['data']['process_HT_use'].max())
        SolarT_HT_use.append(scenarios[n]['data']['SolarT_HT_use'].max())
        GSHP_HT_cap.append(scenarios[n]['data']['GSHP_HT'].max())
        GSHP_MT_cap.append(scenarios[n]['data']['GSHP_MT'].max())
        GSHP_LT_cap.append(scenarios[n]['data']['GSHP_LT'].min())
        HT_storage_cap.append(scenarios[n]['data']['HT_storage'].max())
        MT_storage_cap.append(scenarios[n]['data']['MT_storage'].max())

    raw_data = { 'scenario': scenario_names,
            'process_MT_cap': process_MT_use,
            'process_HT_cap': process_HT_use,
            'SolarT_HT_cap': SolarT_HT_use,
            'GSHP_HT_cap': GSHP_HT_cap,
            'GSHP_MT_cap': GSHP_MT_cap,
            'GSHP_LT_cap': GSHP_LT_cap,
            'HT_storage_cap':HT_storage_cap,
            'MT_storage_cap':MT_storage_cap}
    cap_df = pd.DataFrame(raw_data, columns = ['scenario', 'process_MT_cap', 'process_HT_cap', 'SolarT_HT_cap', 'GSHP_HT_cap', 'GSHP_MT_cap',                                                'GSHP_LT_cap', 'HT_storage_cap', 'MT_storage_cap'])
    #make data labels
    data_labels = []
    data_labels.extend(process_MT_use)
    data_labels.extend(process_HT_use)
    data_labels.extend(SolarT_HT_use)
    data_labels.extend(GSHP_HT_cap)
    data_labels.extend(GSHP_MT_cap)
    data_labels.extend(GSHP_LT_cap)
    data_labels.extend(HT_storage_cap)
    data_labels.extend(MT_storage_cap)
    data_labels = map('{0:.0f}'.format,data_labels)
    # Setting the positions and width for the bars
    pos = list(range(len(cap_df)))
    width = 0.25
    spacing = width*(len(cap_df.columns))

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10,10))
    # ['process_MT_cap', 'process_HT_cap', 'SolarT_HT_cap', 'GSHP_HT_cap', 'GSHP_MT_cap', 'GSHP_LT_cap', 'HT_storage_cap', 'MT_storage_cap'
    colours = ['#DD2D4A', '#880D1E', 'yellow', '#F26A8D',  '#F49CBB','#CBEEF3', '#CC2E49', '#CC2E49', '#39BDC6']

    # Create a bar with pre_score data,
    # in position pos,
    for n in range(len(cap_df.columns)-1):
        plt.bar([p*spacing + width*n for p in pos], 
                cap_df.iloc[:,n+1], 
                # of width
                width, 
                # with alpha 0.5
                alpha=1, 
                # with color
                color=colours[n], 
                # with label the first value in first_name
                #label=cap_df['scenario'][n])
               )

    # Set axis labels
    ax.set_ylabel('Capacity (kW)', fontsize=18)
    ax.set_xlabel('Scenario Number', fontsize=18)
    # Set the chart's title
    ax.set_title('Capacity Characteristics of Scenarios', fontsize=20)

    # Set the position of the x ticks
    ax.set_xticks([p*spacing + 3.5 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(cap_df['scenario'])
    plt.tick_params(labelsize=16)
    # create data labels
    rects = ax.patches

    for rect, label in zip(rects, data_labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height, label, ha='center', va='bottom', fontsize=8)

    # Adding the legend and showing the plot
    plt.legend(['Process MT', 'Process HT', 'Solar Thermal HT','GSHP HT','GSHP MT', 'GSHP LT','HT Network Storage', 'MT Network Storage'], loc='upper left')
    plt.savefig(filename)
    plt.close()

def gen_sankey_input(in_scenario, num):
	# function to generate .txt files with input to fancy looking online sankey diagram generator
    # network constants
    params = in_scenario['params']
    hydro_production = params['hydro_production']
    n_hydro = 0.9
    n_HX = 0.9
    
    GSHP_COP_H = params['GSHP_COP_H']
    GSHP_COP_C = params['GSHP_COP_C']
    oil_n_th = params['oil_n_th']
    gas_n_th = params['gas_n_th']
    
    in_uncovered = in_scenario['uncovered_demand']
    dec_GSHP_COP_H = params['dec_GSHP_COP_H']
    dec_GSHP_COP_C = params['dec_GSHP_COP_C']
    in_df = in_scenario['data']
    pump_elec = in_df['LT_pump_elec'].sum() + in_df['MT_pump_elec'].sum() + in_df['HT_pump_elec'].sum()
    
    tf_str = '\'paste diagram code at http://sankeymatic.com/build/\n\n'
    tf_str += '\'primary sources (MWh)\n'
    tf_str += "Gas Boiler [%s] Zone Heating\n" % '{0:.0f}'.format(in_uncovered['ZH_gas'].sum()/1000)
    tf_str += "Oil Boiler [%s] Zone Heating\n" % '{0:.0f}'.format(in_uncovered['ZH_oil'].sum()/1000)
    tf_str += "Gas [%s] Gas Boiler #4c5d70\n" % '{0:.0f}'.format(in_uncovered['ZH_gas'].sum()/gas_n_th/1000)
    tf_str += "Oil [%s] Oil Boiler\n" % '{0:.0f}'.format(in_uncovered['ZH_oil'].sum()/oil_n_th/1000)
    tf_str += "Process Heat [%s] Heat Exchanger\n" % '{0:.0f}'.format((in_df['process_MT_use'].sum()/0.9+in_df['process_HT_use'].sum()/0.9)/1000)
    tf_str += "Biomass Incineration [%s] High-T Network #55b752\n" % '{0:.0f}'.format((in_df['Biomass_HT_use'].sum()/1000))
    tf_str += "Solar Thermal HT [%s] High-T Network\n" % '{0:.0f}'.format((in_df['SolarT_HT_use'].sum()/1000))
    tf_str += "Boreholes [%s] GSHP HT #55b752\n" % '{0:.0f}'.format((in_df['GSHP_HT'].sum()*(1-1/GSHP_COP_H))/1000)
    tf_str += "Boreholes [%s] GSHP MT #55b752\n" % '{0:.0f}'.format((in_df['GSHP_MT'].sum()*(1-1/GSHP_COP_H))/1000)
    tf_str += "Boreholes [%s] GSHP LT #55b752\n" % '{0:.0f}'.format((-in_df['GSHP_LT'].sum()*(1-1/GSHP_COP_C))/1000)
    tf_str += "Electric Grid [%s] GSHP HT #ffdf00\n" % '{0:.0f}'.format((in_df['GSHP_HT'].sum()/GSHP_COP_H)/1000)
    tf_str += "Electric Grid [%s] GSHP MT #ffdf00\n" % '{0:.0f}'.format((in_df['GSHP_MT'].sum()/GSHP_COP_H)/1000)
    tf_str += "Electric Grid [%s] GSHP LT #ffdf00\n" % '{0:.0f}'.format((-in_df['GSHP_LT'].sum()/GSHP_COP_C)/1000)
    tf_str += "Electric Grid [%s] DHW #ffdf00\n" % '{0:.0f}'.format(in_uncovered['DHW'].sum()/1000)
    tf_str += "Electric Grid [%s] Building GSHP #ffdf00\n" % '{0:.0f}'.format((in_uncovered['ZH_GSHP'].sum()/dec_GSHP_COP_H)/1000)
    tf_str += "Building GSHP [%s] Zone Heating #ffdf00\n" % '{0:.0f}'.format((in_uncovered['ZH_GSHP'].sum())/1000)
    tf_str += "Electric Grid [%s] Building Chiller #ffdf00\n" % '{0:.0f}'.format((-in_uncovered['Total Cooling'].sum()/dec_GSHP_COP_C)/1000)
    tf_str += "Building Chiller [%s] Zone Cooling #3EBBC4\n" % '{0:.0f}'.format((-in_uncovered['Total Cooling'].sum())/1000)
    tf_str += "Sihl [%s] Hydro Plant\n" % '{0:.0f}'.format(hydro_production/n_hydro/1000)
    tf_str += '\'Secondary sources (MWh)\n'
    tf_str += 'GSHP HT [%s] High-T Network\n' % '{0:.0f}'.format((in_df['GSHP_HT'].sum())/1000)
    tf_str += 'GSHP MT [%s] Mid-T Network\n' % '{0:.0f}'.format((in_df['GSHP_MT'].sum())/1000)
    tf_str += 'GSHP LT [%s] Low-T Network\n' % '{0:.0f}'.format((-in_df['GSHP_LT'].sum())/1000)
    tf_str += 'High-T Network [%s] Mid-T Network\n' % '{0:.0f}'.format(in_df['HT_to_MT'].sum()/1000)
    tf_str += 'Hydro Plant [%s] Electric Gen.\n' % '{0:.0f}'.format(hydro_production/1000)
    tf_str += 'Heat Exchanger [%s] Mid-T Network\n' % '{0:.0f}'.format(in_df['process_MT_use'].sum()/1000)
    tf_str += 'Heat Exchanger [%s] High-T Network\n' % '{0:.0f}'.format(in_df['process_HT_use'].sum()/1000)
    tf_str += '\'Distribution (MWh)\n'
    tf_str += 'High-T Network [%s] DHW\n' % '{0:.0f}'.format((in_df['DHW'].sum()-in_df['DHW_preheat'].sum())/1000)
    tf_str += 'High-T Network [%s] HT Losses\n' % '{0:.0f}'.format((in_df['HT_loss'].sum())/1000)
    #tf_str += 'High-T Network [%s] HT Export\n' % '{0:.0f}'.format((in_df['HT_export'].sum())/1000)
    tf_str += 'Mid-T Network [%s] DHW\n' % '{0:.0f}'.format((in_df['DHW_preheat'].sum())/1000)
    tf_str += 'Mid-T Network [%s] Zone Heating\n' % '{0:.0f}'.format((in_df['Zone Heating'].sum())/1000)
    tf_str += 'Mid-T Network [%s] MT Losses\n' % '{0:.0f}'.format((in_df['MT_loss'].sum())/1000)
    #tf_str += 'Mid-T Network [%s] MT Export\n' % '{0:.0f}'.format((in_df['MT_export'].sum())/1000)
    tf_str += 'Low-T Network [%s] Zone Cooling\n' % '{0:.0f}'.format((-in_df['Total Cooling'].sum())/1000)
    tf_str += 'Low-T Network [%s] LT Losses\n' % '{0:.0f}'.format((-in_df['LT_loss'].sum())/1000)
    tf_str += 'Electric Gen. [%s] Pump Energy\n' % '{0:.0f}'.format(pump_elec/1000)
    tf_str += 'Electric Gen. [%s] Grid Feed-in\n' % '{0:.0f}'.format((hydro_production-pump_elec)/1000)
    tf_str += '\' Color settings\n'
    tf_str += ':High-T Network #CC0000\n'
    tf_str += ':Mid-T Network #ff0000\n'
    tf_str += ':Gas #BF3D48\n'
    tf_str += ':DHW #CC0000\n'
    tf_str += ':HT Losses #BF3D48\n'
    tf_str += ':Process Heat #BF3D48\n'
    tf_str += ':Heat Exchanger #BF3D48\n'
    tf_str += ':Zone Heating #ff0000\n'
    tf_str += ':LT Losses #8eabad\n'
    tf_str += ':HT Losses #8eabad\n'
    tf_str += ':Zone Cooling #3EBBC4\n'
    tf_str += ':Electric Gen. #f9ee11\n'
    tf_str += ':Pump Energy #6be050\n'
    tf_str += ':MT Losses #8eabad\n'
    tf_str += ':GSHP MT #ff0000\n'
    tf_str += ':GSHP LT #2e70ba\n'
    tf_str += ':Boreholes #55b752\n'
    tf_str += ':Gas #4c5d70\n'
    tf_str += ':Low-T Network #2e70ba\n'
    tf_str += ':Electric Grid #ffdf00\n'
    tf_str += ':Grid Feed-in #ffdf00\n'
    tf_str += ':Oil #49231b\n'
    tf_str += ':Solar Thermal HT #BF3D48\n'
    tf_str += ':Oil Boiler #49231b\n'
    tf_str += ':Biomass Incineration #55b752\n'
    tf_str += ':Building Chiller #3EBBC4\n'
    tf_str += ':Gas Boiler #ff0000\n'
    tf_str += ':Building GSHP #ff0000\n'
    
    text_file = open('v3_sankey_'+ str(num) +'.txt', 'w')
    text_file.write(tf_str)
    text_file.close()

def plot_scenario_COP(scenarios, filename):
      
    # network constants

    scenario_names = []
    ZH_COP = []
    DHW_COP = []
    Cooling_COP = []
    GSHP_COP_H = []
    GSHP_COP_C = []

    for n in range(len(scenarios)):
        scenario_names.append(n)
        params = scenarios[n]['params']
        GSHP_COP_H = params['GSHP_COP_H']
        GSHP_COP_C = params['GSHP_COP_C']
        ZH_COP.append(scenarios[n]['data']['Zone Heating'].sum()/(((scenarios[n]['data']['GSHP_MT'].sum()/GSHP_COP_H)*                         (scenarios[n]['data']['Zone Heating'].sum()/                          (scenarios[n]['data']['Zone Heating'].sum()+scenarios[n]['data']['DHW_preheat'].sum())))                        +scenarios[n]['data']['MT_pump_elec'].sum()))
        DHW_COP.append(scenarios[n]['data']['DHW'].sum()/((((scenarios[n]['data']['GSHP_HT'].sum()/GSHP_COP_H) +                        ((scenarios[n]['data']['GSHP_MT'].sum()/GSHP_COP_H)*                         (scenarios[n]['data']['DHW_preheat'].sum()/                          (scenarios[n]['data']['Zone Heating'].sum()+scenarios[n]['data']['DHW_preheat'].sum()))))                       +scenarios[n]['data']['HT_pump_elec'].sum())))
        Cooling_COP.append((scenarios[n]['data']['Total Cooling'].sum())/(scenarios[n]['data']['GSHP_LT'].sum()/GSHP_COP_C                         -scenarios[n]['data']['LT_pump_elec'].sum()))

    raw_data = { 'scenario': scenario_names,
            'ZH_COP': ZH_COP,
            'DHW_COP': DHW_COP,
            'Cooling_COP': Cooling_COP
               }
    cap_df = pd.DataFrame(raw_data, columns = ['scenario', 'ZH_COP', 'DHW_COP', 'Cooling_COP'])
    #make data labels
    data_labels = []
    data_labels.extend(ZH_COP)
    data_labels.extend(DHW_COP)
    data_labels.extend(Cooling_COP)
    data_labels = map('{0:.1f}'.format,data_labels)
    # Setting the positions and width for the bars
    pos = list(range(len(cap_df)))
    width = 0.25
    spacing = width*(len(cap_df.columns))

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(20,10))
    # ['ZH_COP', 'DHW_COP', 'Cooling_COP']
    colours = ['orange', 'red', 'blue']

    # Create a bar with pre_score data,
    # in position pos,
    for n in range(len(cap_df.columns)-1):
        plt.bar([p*spacing + width*n for p in pos], 
                cap_df.iloc[:,n+1], 
                # of width
                width,
                # with alpha 0.5
                alpha=1.0, 
                # with color
                color=colours[n], 
                # with label the first value in first_name
                #label=cap_df['scenario'][n])
               )

    # Set axis labels
    ax.set_ylabel('Heat-to-power ratio (kWh-th/kWh-elec)', fontsize=18)
    ax.set_xlabel('Scenario Number', fontsize=18)
    # Set the chart's title
    ax.set_title('Heat-to-power ratio for all Scenarios', fontsize=20)
    # Set the position of the x ticks
    ax.set_xticks([p*spacing + 1 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(cap_df['scenario'])
    plt.tick_params(labelsize=16)
    # Now make some labels
    rects = ax.patches
    for rect, label in zip(rects, data_labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 0.1, label, ha='center', va='bottom', fontsize=16)
    
    # Adding the legend and showing the plot
    plt.legend(['Zone Heating', 'DHW', 'Cooling'], loc='upper left')
    plt.savefig(filename)
    plt.close()

def plot_scenarios_CO2(scenarios, filename):
    # compute and plot tonnes-CO2/kWh for each network in each scenario  
    # network constants

    scenario_names = []
    ZH_CO2 = []
    DHW_CO2 = []
    Cooling_CO2 = []
    GSHP_COP_H = []
    GSHP_COP_C = []

    for n in range(len(scenarios)):
        scenario_names.append(n)
        params = scenarios[n]['params']
        GSHP_COP_H = params['GSHP_COP_H']
        GSHP_COP_C = params['GSHP_COP_C']
        Elec_CO2_kWh = params['Elec_CO2_kWh']
        Gas_CO2_kWh = params['Gas_CO2_kWh']
        ZH_CO2_kg = ((scenarios[n]['data']['GSHP_MT'].sum()/GSHP_COP_H)*(scenarios[n]['data']['Zone Heating'].sum()/                     (scenarios[n]['data']['Zone Heating'].sum()+scenarios[n]['data']['DHW_preheat'].sum())))*Elec_CO2_kWh
        ZH_demand = scenarios[n]['data']['Zone Heating'].sum()
        ZH_CO2.append(ZH_CO2_kg/ZH_demand)
        DHW_CO2_kg = ((scenarios[n]['data']['GSHP_HT'].sum()/GSHP_COP_H)+(scenarios[n]['data']['GSHP_MT'].sum()/GSHP_COP_H)*                     (scenarios[n]['data']['DHW_preheat'].sum()/                     (scenarios[n]['data']['Zone Heating'].sum()+scenarios[n]['data']['DHW_preheat'].sum())))*Elec_CO2_kWh
        DHW_demand = scenarios[n]['data']['DHW'].sum()
        DHW_CO2.append(DHW_CO2_kg/DHW_demand)  
        Cooling_CO2_kg = (-scenarios[n]['data']['GSHP_LT'].sum()/GSHP_COP_C)*Elec_CO2_kWh
        Cooling_demand = -scenarios[n]['data']['Total Cooling'].sum()
        Cooling_CO2.append(Cooling_CO2_kg/Cooling_demand)                    
    
    raw_data = { 'scenario': scenario_names,
            'ZH_CO2': ZH_CO2,
            'DHW_CO2': DHW_CO2,
            'Cooling_CO2': Cooling_CO2
               }
    cap_df = pd.DataFrame(raw_data, columns = ['scenario', 'ZH_CO2', 'DHW_CO2', 'Cooling_CO2'])

    #make data labels
    data_labels = []
    data_labels.extend(ZH_CO2)
    data_labels.extend(DHW_CO2)
    data_labels.extend(Cooling_CO2)
    data_labels = map('{0:.2f}'.format,data_labels)
    # Setting the positions and width for the bars
    pos = list(range(len(cap_df)))
    width = 0.25
    spacing = width*(len(cap_df.columns))

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(20,10))
    # ['ZH_COP', 'DHW_COP', 'Cooling_COP']
    colours = ['orange', 'red', 'blue']

    # Create a bar with pre_score data,
    # in position pos,
    for n in range(len(cap_df.columns)-1):
        plt.bar([p*spacing + width*n for p in pos], 
                cap_df.iloc[:,n+1], 
                # of width
                width,
                # with alpha 0.5
                alpha=1.0, 
                # with color
                color=colours[n], 
                # with label the first value in first_name
                #label=cap_df['scenario'][n])
               )

    # Set axis labels
    ax.set_ylabel('Carbon Intensity (kg-CO2/kWh)', fontsize=18)
    ax.set_xlabel('Scenario Number', fontsize=18)
    # Set the chart's title
    ax.set_title('Network CO2 Intensity for Scenarios', fontsize=20)
    # Set the position of the x ticks
    ax.set_xticks([p*spacing + 1 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(cap_df['scenario'])
    plt.tick_params(labelsize=16)
    # Now make some labels
    rects = ax.patches
    for rect, label in zip(rects, data_labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height, label, ha='center', va='bottom', fontsize=16)

    # Adding the legend and showing the plot
    plt.legend(['Zone Heating CO2', 'DHW CO2', 'Cooling CO2'], loc='upper right')
    plt.savefig(filename)
    plt.close()

def plot_district_CO2(scenarios, filename):
    # compute and plot tonnes-CO2/kWh for each network in each scenario  
    scenario_names = []
    ZH_CO2 = []
    DHW_CO2 = []
    Cooling_CO2 = []
    # network params
    GSHP_COP_H = 0
    GSHP_COP_C = 0
    oil_n_th = 0
    gas_n_th = 0
    dec_GSHP_COP_H = 0
    dec_GSHP_COP_C = 0

    for n in range(len(scenarios)):
        scenario_names.append(n)
        params = scenarios[n]['params']
        GSHP_COP_H = params['GSHP_COP_H']
        GSHP_COP_C = params['GSHP_COP_C']
        oil_n_th = params['oil_n_th']
        gas_n_th = params['gas_n_th']
        dec_GSHP_COP_H = params['dec_GSHP_COP_H']
        dec_GSHP_COP_C = params['dec_GSHP_COP_C']
        Elec_CO2_kWh = params['Elec_CO2_kWh']
        Gas_CO2_kWh = params['Gas_CO2_kWh']
        Oil_CO2_kWh = params['Oil_CO2_kWh']
        
        ZH_CO2_kg = ((scenarios[n]['data']['GSHP_MT'].sum()/GSHP_COP_H)*(scenarios[n]['data']['Zone Heating'].sum()/(scenarios[n]['data']['Zone Heating'].sum()+scenarios[n]['data']['DHW_preheat'].sum())))*Elec_CO2_kWh +                    (scenarios[n]['uncovered_demand']['ZH_oil'].sum()/oil_n_th)*Oil_CO2_kWh +                    (scenarios[n]['uncovered_demand']['ZH_gas'].sum()/gas_n_th)*Gas_CO2_kWh +                    (scenarios[n]['uncovered_demand']['ZH_GSHP'].sum()/dec_GSHP_COP_H)*Elec_CO2_kWh
        ZH_demand = scenarios[n]['data']['Zone Heating'].sum() + scenarios[n]['uncovered_demand']['ZH_oil'].sum() +scenarios[n]['uncovered_demand']['ZH_gas'].sum() +                     scenarios[n]['uncovered_demand']['ZH_GSHP'].sum()
        ZH_CO2.append(ZH_CO2_kg/ZH_demand)
        
        DHW_CO2_kg = ((scenarios[n]['data']['GSHP_HT'].sum()/GSHP_COP_H)+(scenarios[n]['data']['GSHP_MT'].sum()/GSHP_COP_H)*(scenarios[n]['data']['DHW_preheat'].sum()/                      (scenarios[n]['data']['Zone Heating'].sum()+scenarios[n]['data']['DHW_preheat'].sum())))*Elec_CO2_kWh +                     (scenarios[n]['uncovered_demand']['DHW'].sum())*Elec_CO2_kWh
        DHW_demand = scenarios[n]['data']['DHW'].sum() + scenarios[n]['uncovered_demand']['DHW'].sum()
        DHW_CO2.append(DHW_CO2_kg/DHW_demand)
        
        Cooling_CO2_kg = (-scenarios[n]['data']['GSHP_LT'].sum()/GSHP_COP_C)*Elec_CO2_kWh + (-scenarios[n]['uncovered_demand']['Total Cooling'].sum()/dec_GSHP_COP_C)*Elec_CO2_kWh
        Cooling_demand = -scenarios[n]['data']['Total Cooling'].sum() + -scenarios[n]['uncovered_demand']['Total Cooling'].sum()
        Cooling_CO2.append(Cooling_CO2_kg/Cooling_demand)                    
    
    raw_data = { 'scenario': scenario_names,
            'ZH_CO2': ZH_CO2,
            'DHW_CO2': DHW_CO2,
            'Cooling_CO2': Cooling_CO2
               }
    cap_df = pd.DataFrame(raw_data, columns = ['scenario', 'ZH_CO2', 'DHW_CO2', 'Cooling_CO2'])

    #make data labels
    data_labels = []
    data_labels.extend(ZH_CO2)
    data_labels.extend(DHW_CO2)
    data_labels.extend(Cooling_CO2)
    data_labels = map('{0:.2f}'.format,data_labels)
    # Setting the positions and width for the bars
    pos = list(range(len(cap_df)))
    width = 0.25
    spacing = width*(len(cap_df.columns))

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(20,10))
    # ['ZH_COP', 'DHW_COP', 'Cooling_COP']
    colours = ['orange', 'red', 'blue']

    # Create a bar with pre_score data,
    # in position pos,
    for n in range(len(cap_df.columns)-1):
        plt.bar([p*spacing + width*n for p in pos], 
                cap_df.iloc[:,n+1], 
                # of width
                width,
                # with alpha 0.5
                alpha=1.0, 
                # with color
                color=colours[n], 
                # with label the first value in first_name
                #label=cap_df['scenario'][n])
               )

    # Set axis labels
    ax.set_ylabel('Carbon Intensity (kg-CO2/kWh)', fontsize=18)
    ax.set_xlabel('Scenario Number', fontsize=18)
    # Set the chart's title
    ax.set_title('District CO2 Intensity for Scenarios', fontsize=20)
    # Set the position of the x ticks
    ax.set_xticks([p*spacing + 1 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(cap_df['scenario'])
    plt.tick_params(labelsize=16)
    # Now make some labels
    rects = ax.patches
    for rect, label in zip(rects, data_labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height, label, ha='center', va='bottom', fontsize=16)

    # Adding the legend and showing the plot
    plt.legend(['Zone Heating CO2', 'DHW CO2', 'Cooling CO2'], loc='upper right')
    plt.savefig(filename)
    plt.close()

def plot_process_use(scenarios, filename):

    scenario_names = []
    MT_process_factor = []
    HT_process_factor = []
    HT_SolarT_factor = []
    HT_biomass_factor = []

    for n in range(len(scenarios)):
        scenario_names.append(n)
        
        if scenarios[n]['data']['process_MT_use'].sum() > 0:
            MT_process_factor.append(scenarios[n]['data']['process_MT_use'].sum()/scenarios[n]['data']['process_MT_gen'].sum())
        else:
            MT_process_factor.append(0)
        if scenarios[n]['data']['process_HT_use'].sum() > 0:
            HT_process_factor.append(scenarios[n]['data']['process_HT_use'].sum()/scenarios[n]['data']['process_HT_gen'].sum())           
        else:
            HT_process_factor.append(0)
        if scenarios[n]['data']['SolarT_HT_use'].sum() > 0:
            HT_SolarT_factor.append(scenarios[n]['data']['SolarT_HT_use'].sum()/scenarios[n]['data']['SolarT_HT_gen'].sum())           
        else:
            HT_SolarT_factor.append(0)
        if scenarios[n]['data']['Biomass_HT_use'].sum() > 0:
            HT_biomass_factor.append(scenarios[n]['data']['Biomass_HT_use'].sum()/scenarios[n]['data']['Biomass_HT_gen'].sum())           
        else:
            HT_biomass_factor.append(0)
            
    raw_data = { 'scenario': scenario_names,
            'MT_process_factor': MT_process_factor,
            'HT_process_factor': HT_process_factor,
            'HT_SolarT_factor': HT_SolarT_factor,
            'HT_biomass_factor': HT_biomass_factor,
               }
    cap_df = pd.DataFrame(raw_data, columns = ['scenario', 'MT_process_factor', 'HT_process_factor', 'HT_SolarT_factor', 'HT_biomass_factor'])
    
    #make data labels
    data_labels = []
    data_labels.extend(MT_process_factor)
    data_labels.extend(HT_process_factor)
    data_labels.extend(HT_SolarT_factor)
    data_labels.extend(HT_biomass_factor)
    data_labels = map('{0:.2f}'.format,data_labels)
    # Setting the positions and width for the bars
    pos = list(range(len(cap_df)))
    width = 0.25
    spacing = width*(len(cap_df.columns))

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(20,10))
    # ['ZH_COP', 'DHW_COP', 'Cooling_COP']
    colours = ['orange', 'red', 'yellow', 'brown']

    # Create a bar with pre_score data,
    # in position pos,
    for n in range(len(cap_df.columns)-1):
        plt.bar([p*spacing + width*n for p in pos], 
                cap_df.iloc[:,n+1], 
                # of width
                width,
                # with alpha 0.5
                alpha=1.0, 
                # with color
                color=colours[n], 
                # with label the first value in first_name
                #label=cap_df['scenario'][n])
               )

    # Set axis labels
    ax.set_ylabel('Heat recovery and use factor', fontsize=18)
    ax.set_xlabel('Scenario Number', fontsize=18)
    # Set the chart's title
    ax.set_title('Heat recovery and use factor for all Scenarios', fontsize=20)
    # Set the position of the x ticks
    ax.set_xticks([p*spacing + 1 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(cap_df['scenario'])
    plt.tick_params(labelsize=16)
    # Now make some labels
    rects = ax.patches
    for rect, label in zip(rects, data_labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height, label, ha='center', va='bottom', fontsize=16)
    
    # Adding the legend and showing the plot
    plt.legend(['MT Process Heat Recovery', 'HT Process Heat Recovery', 'HT Solar Thermal Use Factor', 'HT Biomass Use Factor'], loc='upper left')
    plt.savefig(filename)
    plt.close()

def plot_exports(scenarios, filename):

    scenario_names = []
    exports = []

    for n in range(len(scenarios)):
        scenario_names.append(n)
        exports.append((scenarios[n]['data']['HT_export'].sum()+ scenarios[n]['data']['MT_export'].sum())/1000)       
            
    raw_data = { 'scenario': scenario_names,
            'exports': exports,
               }
    cap_df = pd.DataFrame(raw_data, columns = ['scenario', 'exports'])
    
    #make data labels
    data_labels = []
    data_labels.extend(exports)
    data_labels = map('{0:.0f}'.format,data_labels)
    # Setting the positions and width for the bars
    pos = list(range(len(cap_df)))
    width = 0.25
    spacing = width*(len(cap_df.columns))

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(20,10))
    # ['ZH_COP', 'DHW_COP', 'Cooling_COP']
    colours = ['red', 'yellow', 'brown']

    # Create a bar with pre_score data,
    # in position pos,
    for n in range(len(cap_df.columns)-1):
        plt.bar([p*spacing + width*n for p in pos], 
                cap_df.iloc[:,n+1], 
                # of width
                width,
                # with alpha 0.5
                alpha=1.0, 
                # with color
                color=colours[n], 
                # with label the first value in first_name
                #label=cap_df['scenario'][n])
               )

    # Set axis labels
    ax.set_ylabel('Thermal Exports (MWh)', fontsize=18)
    ax.set_xlabel('Scenario Number', fontsize=18)
    # Set the chart's title
    ax.set_title('Thermal Exports to Interdistrict Heating Network', fontsize=20)
    # Set the position of the x ticks
    ax.set_xticks([p*spacing + 0 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(cap_df['scenario'])
    plt.tick_params(labelsize=16)
    # Now make some labels
    rects = ax.patches
    for rect, label in zip(rects, data_labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height, label, ha='center', va='bottom', fontsize=16)

    # Adding the legend and showing the plot
    #plt.legend(['MT Exports', 'HT Exports'], loc='upper left')
    plt.savefig(filename)
    plt.close()

def plot_2000W_watt_goals(scenarios, filename):
    # compute and plot tonnes-CO2/kWh for each network in each scenario  
    scenario_names = []
    ZH_CO2 = []
    DHW_CO2 = []
    Cooling_CO2 = []
    # network params
    GSHP_COP_H = 0
    GSHP_COP_C = 0
    oil_n_th = 0
    gas_n_th = 0
    dec_GSHP_COP_H = 0
    dec_GSHP_COP_C = 0
    
    scenario_names = []
    Heating_watt_p_person = []
    Elec_watt_p_person = []
    persons = 2500
    
    for n in range(len(scenarios)):
        scenario_names.append(n)
        params = scenarios[n]['params']
        GSHP_COP_H = params['GSHP_COP_H']
        GSHP_COP_C = params['GSHP_COP_C']
        oil_n_th = params['oil_n_th']
        gas_n_th = params['gas_n_th']
        dec_GSHP_COP_H = params['dec_GSHP_COP_H']
        dec_GSHP_COP_C = params['dec_GSHP_COP_C']
        Elec_CO2_kWh = params['Elec_CO2_kWh']
        Gas_CO2_kWh = params['Gas_CO2_kWh']
        Oil_CO2_kWh = params['Oil_CO2_kWh']
        
        ZH_CO2_kg = ((scenarios[n]['data']['GSHP_MT'].sum()/GSHP_COP_H)* (scenarios[n]['data']['Zone Heating'].sum()/                     (scenarios[n]['data']['Zone Heating'].sum()+scenarios[n]['data']['DHW_preheat'].sum())))*Elec_CO2_kWh +                    (scenarios[n]['uncovered_demand']['ZH_oil'].sum()/oil_n_th)*Oil_CO2_kWh +                    (scenarios[n]['uncovered_demand']['ZH_gas'].sum()/gas_n_th)*Gas_CO2_kWh +                    (scenarios[n]['uncovered_demand']['ZH_GSHP'].sum()/dec_GSHP_COP_H)*Elec_CO2_kWh
        ZH_watt = ((scenarios[n]['data']['GSHP_MT'].sum()/GSHP_COP_H)* (scenarios[n]['data']['Zone Heating'].sum()/                     (scenarios[n]['data']['Zone Heating'].sum()+scenarios[n]['data']['DHW_preheat'].sum()))) +                    (scenarios[n]['uncovered_demand']['ZH_oil'].sum()/oil_n_th) +                    (scenarios[n]['uncovered_demand']['ZH_gas'].sum()/gas_n_th) +                    (scenarios[n]['uncovered_demand']['ZH_GSHP'].sum()/dec_GSHP_COP_H)
        
        DHW_CO2_kg = ((scenarios[n]['data']['GSHP_HT'].sum()/GSHP_COP_H)+(scenarios[n]['data']['GSHP_MT'].sum()/GSHP_COP_H)*                      (scenarios[n]['data']['DHW_preheat'].sum()/                      (scenarios[n]['data']['Zone Heating'].sum()+scenarios[n]['data']['DHW_preheat'].sum())))*Elec_CO2_kWh +                     (scenarios[n]['uncovered_demand']['DHW'].sum())*Elec_CO2_kWh
        
        DHW_watt = ((scenarios[n]['data']['GSHP_HT'].sum()/GSHP_COP_H)+(scenarios[n]['data']['GSHP_MT'].sum()/GSHP_COP_H)*                      (scenarios[n]['data']['DHW_preheat'].sum()/                      (scenarios[n]['data']['Zone Heating'].sum()+scenarios[n]['data']['DHW_preheat'].sum()))) +                     (scenarios[n]['uncovered_demand']['DHW'].sum())
        
        Cooling_CO2_kg = (-scenarios[n]['data']['GSHP_LT'].sum()/GSHP_COP_C)*Elec_CO2_kWh + (-scenarios[n]['uncovered_demand']['Total Cooling'].sum()/dec_GSHP_COP_C)*Elec_CO2_kWh
        Cooling_watt = (-scenarios[n]['data']['GSHP_LT'].sum()/GSHP_COP_C)+ (-scenarios[n]['uncovered_demand']['Total Cooling'].sum()/dec_GSHP_COP_C)
        
        Elec_watt = scenarios[n]['data']['Room Electricity'].sum()+scenarios[n]['uncovered_demand']['Room Electricity'].sum()
        Elec_CO2_kg = (scenarios[n]['data']['Room Electricity'].sum()+scenarios[n]['uncovered_demand']['Room Electricity'].sum())*Elec_CO2_kWh
        Lighting_watt = (scenarios[n]['data']['Lighting'].sum()+scenarios[n]['uncovered_demand']['Lighting'].sum())
        Lighting_CO2_kg = (scenarios[n]['data']['Lighting'].sum()+scenarios[n]['uncovered_demand']['Lighting'].sum())*Elec_CO2_kWh
        
        Heating_watt_p_person.append((ZH_watt+DHW_watt+Cooling_watt)*1000/(persons*8760))
        Elec_watt_p_person.append((Elec_watt+Lighting_watt)*1000/(persons*8760))

    
    raw_data = { 'scenario': scenario_names,
            'Heating_watt_p_person': Heating_watt_p_person,
            'Elec_watt_p_person': Elec_watt_p_person,
               }
    cap_df = pd.DataFrame(raw_data, columns = ['scenario', 'Heating_watt_p_person',  'Elec_watt_p_person'])
    #make data labels
    data_labels = []
    data_labels.extend(Heating_watt_p_person)
    data_labels.extend(Elec_watt_p_person)
    data_labels = map('{0:.2f}'.format,data_labels)
    # Setting the positions and width for the bars
    pos = list(range(len(cap_df)))
    width = 0.25
    spacing = width*(len(cap_df.columns))

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(20,10))
    colours = ['red', 'yellow']

    # Create a bar with pre_score data,
    # in position pos,
    for n in range(len(cap_df.columns)-1):
        plt.bar([p*spacing + width*n for p in pos], 
                cap_df.iloc[:,n+1], 
                # of width
                width,
                # with alpha 0.5
                alpha=1.0, 
                # with color
                color=colours[n], 
                # with label the first value in first_name
                #label=cap_df['scenario'][n])
               )

    # Set axis labels
    ax.set_ylabel('Annual Watts per Person', fontsize=18)
    ax.set_xlabel('Scenario Number', fontsize=18)
    # Set the chart's title
    ax.set_title('2000 Watt Society Goals - Watts per Person', fontsize=20)
    # Set the position of the x ticks
    ax.set_xticks([p*spacing + 1 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(cap_df['scenario'])
    plt.tick_params(labelsize=16)
    # Now make some labels
    rects = ax.patches
    for rect, label in zip(rects, data_labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height, label, ha='center', va='bottom', fontsize=16)
    
    # Setting the x-axis and y-axis limits
    #plt.xlim(min(pos)-width, max(pos)+width)
    plt.ylim([0,2000] )

    # Adding the legend and showing the plot
    plt.legend(['Heating and Cooling', 'Room Electricity and Lighting'], loc='upper right')
    plt.savefig(filename)
    plt.close()

def plot_2000W_co2_goals(scenarios, filename):
    # compute and plot tonnes-CO2/kWh for each network in each scenario  
    scenario_names = []
    ZH_CO2 = []
    DHW_CO2 = []
    Cooling_CO2 = []
    # network params
    GSHP_COP_H = 0
    GSHP_COP_C = 0
    oil_n_th = 0
    gas_n_th = 0
    dec_GSHP_COP_H = 0
    dec_GSHP_COP_C = 0
    
    scenario_names = []
    Heating_CO2_p_person = []
    Lighting_CO2_p_person = []
    persons = 2500
    
    for n in range(len(scenarios)):
        scenario_names.append(n)
        params = scenarios[n]['params']
        GSHP_COP_H = params['GSHP_COP_H']
        GSHP_COP_C = params['GSHP_COP_C']
        oil_n_th = params['oil_n_th']
        gas_n_th = params['gas_n_th']
        dec_GSHP_COP_H = params['dec_GSHP_COP_H']
        dec_GSHP_COP_C = params['dec_GSHP_COP_C']
        Elec_CO2_kWh = params['Elec_CO2_kWh']
        Gas_CO2_kWh = params['Gas_CO2_kWh']
        Oil_CO2_kWh = params['Oil_CO2_kWh']
        
        ZH_CO2_kg = ((scenarios[n]['data']['GSHP_MT'].sum()/GSHP_COP_H)* (scenarios[n]['data']['Zone Heating'].sum()/                     (scenarios[n]['data']['Zone Heating'].sum()+scenarios[n]['data']['DHW_preheat'].sum())))*Elec_CO2_kWh +                    (scenarios[n]['uncovered_demand']['ZH_oil'].sum()/oil_n_th)*Oil_CO2_kWh +                    (scenarios[n]['uncovered_demand']['ZH_gas'].sum()/gas_n_th)*Gas_CO2_kWh +                    (scenarios[n]['uncovered_demand']['ZH_GSHP'].sum()/dec_GSHP_COP_H)*Elec_CO2_kWh
        ZH_watt = ((scenarios[n]['data']['GSHP_MT'].sum()/GSHP_COP_H)* (scenarios[n]['data']['Zone Heating'].sum()/                     (scenarios[n]['data']['Zone Heating'].sum()+scenarios[n]['data']['DHW_preheat'].sum()))) +                    (scenarios[n]['uncovered_demand']['ZH_oil'].sum()/oil_n_th) +                    (scenarios[n]['uncovered_demand']['ZH_gas'].sum()/gas_n_th) +                    (scenarios[n]['uncovered_demand']['ZH_GSHP'].sum()/dec_GSHP_COP_H)
        
        DHW_CO2_kg = ((scenarios[n]['data']['GSHP_HT'].sum()/GSHP_COP_H)+(scenarios[n]['data']['GSHP_MT'].sum()/GSHP_COP_H)*                      (scenarios[n]['data']['DHW_preheat'].sum()/                      (scenarios[n]['data']['Zone Heating'].sum()+scenarios[n]['data']['DHW_preheat'].sum())))*Elec_CO2_kWh +                     (scenarios[n]['uncovered_demand']['DHW'].sum())*Elec_CO2_kWh
        
        DHW_watt = ((scenarios[n]['data']['GSHP_HT'].sum()/GSHP_COP_H)+(scenarios[n]['data']['GSHP_MT'].sum()/GSHP_COP_H)*                      (scenarios[n]['data']['DHW_preheat'].sum()/                      (scenarios[n]['data']['Zone Heating'].sum()+scenarios[n]['data']['DHW_preheat'].sum()))) +                     (scenarios[n]['uncovered_demand']['DHW'].sum())
        
        Cooling_CO2_kg = (-scenarios[n]['data']['GSHP_LT'].sum()/GSHP_COP_C)*Elec_CO2_kWh + (-scenarios[n]['uncovered_demand']['Total Cooling'].sum()/dec_GSHP_COP_C)*Elec_CO2_kWh
        Cooling_watt = (-scenarios[n]['data']['GSHP_LT'].sum()/GSHP_COP_C)+ (-scenarios[n]['uncovered_demand']['Total Cooling'].sum()/dec_GSHP_COP_C)
        
        Elec_watt = scenarios[n]['data']['Room Electricity'].sum()+scenarios[n]['uncovered_demand']['Room Electricity'].sum()
        Elec_CO2_kg = (scenarios[n]['data']['Room Electricity'].sum()+scenarios[n]['uncovered_demand']['Room Electricity'].sum())*Elec_CO2_kWh
        Lighting_watt = (scenarios[n]['data']['Lighting'].sum()+scenarios[n]['uncovered_demand']['Lighting'].sum())
        Lighting_CO2_kg = (scenarios[n]['data']['Lighting'].sum()+scenarios[n]['uncovered_demand']['Lighting'].sum())*Elec_CO2_kWh
        
        Heating_CO2_p_person.append((ZH_CO2_kg+DHW_CO2_kg+Cooling_CO2_kg)/persons)
        Lighting_CO2_p_person.append((Elec_CO2_kWh+Lighting_CO2_kg)/persons)
    
    raw_data = { 'scenario': scenario_names,
            'Heating_CO2_p_person': Heating_CO2_p_person,
            'Lighting_CO2_p_person': Lighting_CO2_p_person,
               }
    cap_df = pd.DataFrame(raw_data, columns = ['scenario', 'Heating_CO2_p_person',  'Lighting_CO2_p_person'])
    #make data labels
    data_labels = []
    data_labels.extend(Heating_CO2_p_person)
    data_labels.extend(Lighting_CO2_p_person)
    data_labels = map('{0:.2f}'.format,data_labels)
    # Setting the positions and width for the bars
    pos = list(range(len(cap_df)))
    width = 0.25
    spacing = width*(len(cap_df.columns))

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(20,10))
    colours = ['red', 'yellow']

    # Create a bar with pre_score data,
    # in position pos,
    for n in range(len(cap_df.columns)-1):
        plt.bar([p*spacing + width*n for p in pos], 
                cap_df.iloc[:,n+1], 
                # of width
                width,
                # with alpha 0.5
                alpha=1.0, 
                # with color
                color=colours[n], 
                # with label the first value in first_name
                #label=cap_df['scenario'][n])
               )

    # Set axis labels
    ax.set_ylabel('Annual kg-CO2 per Person', fontsize=18)
    ax.set_xlabel('Scenario Number', fontsize=18)
    # Set the chart's title
    ax.set_title('2000 Watt Society Goals - CO2 per Person', fontsize=20)
    # Set the position of the x ticks
    ax.set_xticks([p*spacing + 1 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(cap_df['scenario'])
    plt.tick_params(labelsize=16)
    # Now make some labels
    rects = ax.patches
    for rect, label in zip(rects, data_labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height, label, ha='center', va='bottom', fontsize=16)
    
    # Setting the x-axis and y-axis limits
    #plt.xlim(min(pos)-width, max(pos)+width)
    plt.ylim([0,2000] )

    # Adding the legend and showing the plot
    plt.legend(['Heating and Cooling', 'Room Electricity and Lighting'], loc='upper right')
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    time_start = datetime.utcnow()
    print('Started running...')
    runScenarios()
    print('Finished running in [%s]' % str(datetime.utcnow()-time_start))