import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#----------------------------------
def read_clean_csv(file, double_headed = True, x='temperatureC', y='density'):
    '''
    Reads and cleans a csv imported from WebPlotDigitilizer

    file: csv file imported from WebPlotDigitilizer.
    double_headed: weather the csv has more two headers (e.g. 2Kmin and [x,y])
    x: name of the columns that represents the x axis
    y: name of the columns that represents the y axis
    '''
    if double_headed:
        df = pd.read_csv(file, header=[0,1])
        new_cols = [list(tup) for tup in df.columns.values]
        for i in range(len(df.columns.values)):
            if i % 2 != 0:
                new_cols[i][0] = new_cols[i-1][0]
                new_cols[i][1] = y
            else:
                new_cols[i][1] = x
        columns = [tuple(lis) for lis in new_cols]
        df.columns = columns
        df.columns = ['_'.join(col) for col in df.columns.values]
    else:
        df = pd.read_csv(file,header=None)
        df.columns = [x,y]
    df = df.sort_values(by=y, axis=0)
    df.reset_index(inplace=True, drop=True)
    return df

def gomez_hotza(Tmax:float, Tinit:float, beta:float, n:float, A: float, Ea: float, theta0: float, G:float, total_time:float, dt:float, R = 8.31446262, beta_SI = False, as_df=False):
    '''
    Returns a 2D array with the values of time or temperature and the density or porosity of a material in a sintering process
    
    Tmax: maximum temperature in the process, in °C
    Tinit: initial temperature of process, in °C
    beta: heating rate of the process. Either in °C(K)/min or °C(K)/s
    n: mass transport coefficient, varies from 0.0 to 1.0
    A: Gomez-Hotza preexponent factor
    Ea: material's activation energy, in J/mol
    theta0: initial porosity, dimensionless
    G: mean particle size, in m
    total_time: the total time in the process, in s
    dt: size of each time step, in s
    R: molar gas constant
    beta_SI: weather the beta value is expressed in °C(K)/s or not
    xaxis: weather the X axis is 'Temperature' or 'Time'
    yaxis: weather the Y axis is 'Density' or 'Porosity'
    '''

    #Fixed numbers    
    tsteps = int(total_time/dt)
    Tmax += 273
    Tinit += 273
    
    if not beta_SI:
        betaSI = beta/60
    else:
        betaSI = beta
    
    #Initialize arrays
        #Time arrays
    time = np.arange(0, total_time, dt)
    tn = time**(n)
    tc = time
    
        #Temperature arrays
    T = np.ones(tsteps) *Tmax
    T[0] = Tinit
    n1 = np.zeros(tsteps)
    p = -1
    
    for i in range(tsteps):
        T[i] = Tinit + betaSI*time[i]
        n1[i] = np.exp(-Ea/(R*T[i]))
        p += 1

        if T[i] > Tmax:
            T[i] = Tmax
            break #breaks in case the temperature exceeds Tmax
    p -= 1
    #Porosity and density arrays
    theta = np.zeros(tsteps)
    rho = np.ones(tsteps)
    theta[0] = theta0
    rho[0] = 1 - theta0
    for i in range(p):
        theta[i] = theta0 * np.exp(-(R/Ea) * ((A*n)/G) *n1[i]* tn[i])
        rho[i] = 1 - theta[i]

    #After reaching Tmax temperature
    for i in range(p, tsteps):
        theta[i] = theta[p-1] * np.exp(-(A/G) * np.exp(-Ea/(R*T[p])) * (tn[i] - tn[p-1])/T[p-1])
        rho[i] = 1-theta[i]
    
    results = np.zeros((4, tsteps))
    results[0,:] = tc
    results[1,:] = T
    results[2,:] = rho
    results[3,:] = theta

    if as_df:
        results = pd.DataFrame(results.T, columns=['time','temperatureK', 'density', 'porosity'])
        results['time_min'] = results['time']/60
        results['temperatureC'] = results['temperatureK'] - 273
        
    return results

def validation(df, dfX:object, dfY:object, n_values:list, A_values:list, Ea_values:list,  Tmax:float, Tinit:float, beta:float, theta0: float, G:float, R = 8.31446262, beta_SI = False):
    '''
    Applies a 3D grid search to the gomez_hotza() function.
    
    df: DataFrame with the reference values
    n_values: list of values of n to be tested
    A_values: list of values of A to be tested
    Ea_values: list of values of Ea to be tested
    Tmax: maximum temperature in the process, in °C
    Tinit: initial temperature of process, in °C
    beta: heating rate of the process. Either in °C(K)/min or °C(K)/s
    theta0: initial porosity, dimensionless
    G: mean particle size, in m
    R: molar gas constant
    beta_SI: weather the beta value is expressed in °C(K)/s or not
    xaxis: weather the X axis is 'Temperature' or 'Time'
    yaxis: weather the Y axis is 'Density' or 'Porosity'
    '''
    validX = ['temperatureK', 'time', 'temperatureC', 'time_min']
    validY = ['density', 'porosity']

    if dfX not in validX:
        raise ValueError('xaxis: must be one of %r.' % validX)
    if dfY not in validY:
        raise ValueError('yaxis: must be one of %r.' % validY)
    
    if not beta_SI:
        betaSI = beta/60
    else:
        betaSI = beta
    
    total_time = (Tmax - Tinit)/betaSI
    dt = total_time/len(df)
    best_error = float('inf')
    list_of_dictionary = []
    
    for A in A_values:
        for Ea in Ea_values:
            for n in n_values:
                error_df = pd.DataFrame()
                results = gomez_hotza(Tmax=Tmax, Tinit=Tinit,beta=beta, theta0=theta0, G=G, A=A, Ea=Ea, n=n, total_time=total_time, dt=dt, R=R, beta_SI=beta_SI, as_df=True)
                error_df[dfX] = df[dfX]
                error_df['pred_results'] = results[dfY]
                error_df['ref_results'] = df[dfY].sort_values()
                error_df['squared_error'] = (error_df['pred_results'] - error_df['ref_results']) **2
                error = np.mean(error_df['squared_error'])
                rmse = np.sqrt(error)
                list_of_dictionary.append({'A':A, 'Ea':Ea,'n':n, 'RMSE':rmse})
                if rmse < best_error:
                    best_error = rmse
                    best_params = {'A':A, 'Ea':Ea, 'n':n, 'RMSE':rmse}
                    best = error_df
    data = pd.DataFrame(list_of_dictionary, columns=['A', 'Ea', 'n', 'RMSE'])
    print(f'Best params: {best_params}')
    return best, data
#---------------------------------------------------------------------------

duran = read_clean_csv("C:/Users/gupau/Documents/UFSC - PC/Semestre 23.2/TCC/Duran/duran.csv", double_headed=False)

teste, params = validation(duran,
                            dfX='temperatureC',
                            dfY='density',
                            n_values=[0.35,0.4,0.45],
                            A_values=[400,550,600],
                            Ea_values=[110e3,210e3,310e3],
                            Tmax=1386.22,
                            Tinit=406.415,
                            beta=5,
                            theta0=0.596372,
                            G=10e-9
                            )
print(teste)
print(params)

ytzp = gomez_hotza(Tmax=1386.22,
                   Tinit=406.41,
                   beta=5,
                   n=0.435,
                   A=544.5,
                   Ea=206e3,
                   theta0=0.6,
                   G=10e-9,
                   total_time=2e4,
                   dt=5,
                   as_df=True
                   )

sns.lineplot(data=teste, x='temperatureC', y='pred_results', label='Prediction', marker='o',)
sns.lineplot(data=teste, x='temperatureC', y='ref_results', label='Reference', marker='X')
sns.lineplot(data=ytzp, x='temperatureC', y='density', label='teste')
plt.ylabel('Fractional Density')
plt.xlabel('Temperature, C')
plt.legend()
plt.show()
sns.lineplot(data = alumina_df, x='time_min', y='density')
plt.xlabel('Time, min')
plt.ylabel('Density')
plt.show()
